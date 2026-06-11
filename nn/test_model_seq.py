"""Unit tests for the history-transformer net (nn/model_az_seq.py).

Run:  uv run python -m nn.test_model_seq

Covers (plan Phase 3):
  (a) causal mask: future tokens never affect past hidden states
  (b) KV-cache equivalence: encode_prefix + forward_leaf == forward_full +
      readout at the same absolute index, INCLUDING mixed prefix/path lengths
      (and fp16 kv pools) in one batch
  (c) BOS / position alignment: empty history and empty path behave
  (d) token feature parity with the C++ dump
  (e) the whole surface survives torch.jit.script
"""

from __future__ import annotations

import numpy as np
import torch

from nn.model_az_seq import NUM_MOVES, SEQ_CAP, Big2NetSeqAZ, load_token_feats

torch.manual_seed(0)
RNG = np.random.default_rng(0)


def _rand_inputs(B: int):
    hand = (torch.rand(B, 48) < 0.2).float()
    oppmax = (torch.rand(B, 48) < 0.5).float()
    osz = torch.rand(B)
    usz = torch.rand(B)
    otm = (torch.rand(B) < 0.5).float()
    mpts = torch.rand(B)
    opts = torch.rand(B)
    return hand, oppmax, osz, usz, otm, mpts, opts


def _rand_tokens(B: int, T: int) -> torch.Tensor:
    return torch.from_numpy(RNG.integers(0, NUM_MOVES, size=(B, T))).long()


def _full_readout(net, tokens, idx, sides):
    H = net.forward_full(tokens)
    B = tokens.shape[0]
    g = idx.view(B, 1, 1).expand(B, 1, net.d_model)
    h = H.gather(1, g).squeeze(1)
    return net.readout(h, *sides)


def test_token_feats():
    tf = load_token_feats()
    assert tf.shape == (NUM_MOVES, 48)
    assert tf[0].abs().sum() == 0, "pass token must be all-zero"
    # single 3 (move id 1): exactly the count==1 bit of rank 0
    assert tf[1][0] == 1.0 and tf[1].sum() == 1.0
    print("ok test_token_feats")


def test_causal(net):
    B, T = 4, 30
    toks = _rand_tokens(B, T)
    H1 = net.forward_full(toks)
    toks2 = toks.clone()
    toks2[:, 20:] = _rand_tokens(B, T - 20)  # perturb the future
    H2 = net.forward_full(toks2)
    # hidden at sequence index <= 20 (BOS + moves 0..19) must be unchanged
    assert torch.allclose(H1[:, :21], H2[:, :21], atol=1e-6), "causality broken"
    assert not torch.allclose(H1[:, 21:], H2[:, 21:], atol=1e-4), "future inert?!"
    print("ok test_causal")


def test_kv_equivalence(net, fp16_pool: bool = False):
    """Mixed prefix lengths (incl. 0) and path lengths (incl. 0) in ONE batch."""
    B = 6
    plens = [0, 0, 3, 17, 40, 61]  # true history move counts
    tlens = [0, 5, 1, 0, 12, 3]  # in-tree path lengths
    P = max(plens)
    T = max(max(tlens), 1)
    tokens = _rand_tokens(B, P)
    path = _rand_tokens(B, T)
    lens = torch.tensor(plens, dtype=torch.long)
    path_len = torch.tensor(tlens, dtype=torch.long)
    sides = _rand_inputs(B)

    kv, h_last = net.encode_prefix(tokens, lens)
    if fp16_pool:
        kv = kv.half()
    out_leaf = net.forward_leaf(kv, lens + 1, h_last, path, path_len, *sides)

    # Reference: per-row full recompute over concat(prefix, path).
    refs = []
    for b in range(B):
        seq = torch.cat([tokens[b, : plens[b]], path[b, : tlens[b]]]).unsqueeze(0)
        if seq.shape[1] == 0:
            seq = torch.zeros(1, 1, dtype=torch.long)  # dummy; read BOS at idx 0
        idx = torch.tensor([plens[b] + tlens[b]])
        refs.append(_full_readout(net, seq, idx, tuple(s[b : b + 1] for s in sides)))
    atol = 1e-2 if fp16_pool else 1e-4
    for j, name in enumerate(["value", "policy", "behavior", "qa"]):
        ref = torch.cat([r[j] for r in refs], dim=0)
        got = out_leaf[j]
        assert torch.allclose(got, ref, atol=atol), (
            f"kv-vs-full mismatch head={name} fp16={fp16_pool} "
            f"max={float((got - ref).abs().max())}"
        )
    print(f"ok test_kv_equivalence fp16={fp16_pool}")


def test_bos_alignment(net):
    # Empty history: readout at index 0 == BOS hidden; encode_prefix with
    # lens=0 must hand back the same h_last that forward_full produces.
    toks = _rand_tokens(1, 5)
    sides = _rand_inputs(1)
    kv, h_last = net.encode_prefix(toks[:, :0].long().view(1, 0), torch.zeros(1).long())
    H = net.forward_full(toks)
    # BOS hidden from a longer sequence equals h_last (BOS attends only itself)
    assert torch.allclose(H[:, 0], h_last, atol=1e-5), "BOS hidden drifts with tail"
    out_a = net.readout(h_last, *sides)
    out_b = _full_readout(net, toks, torch.zeros(1).long(), sides)
    for a, b in zip(out_a, out_b):
        assert torch.allclose(a, b, atol=1e-5)
    print("ok test_bos_alignment")


def test_scripted_roundtrip(net):
    s = torch.jit.script(net)
    assert s.config() == [
        net.n_layers,
        net.n_heads,
        net.head_dim,
        net.d_model,
        net.seq_cap,
    ]
    B = 3
    toks = _rand_tokens(B, 10)
    lens = torch.tensor([10, 4, 0], dtype=torch.long)
    path = _rand_tokens(B, 4)
    plen = torch.tensor([4, 2, 0], dtype=torch.long)
    sides = _rand_inputs(B)
    kv, h_last = s.encode_prefix(toks, lens)
    a = s.forward_leaf(kv, lens + 1, h_last, path, plen, *sides)
    b = net.forward_leaf(kv, lens + 1, h_last, path, plen, *sides)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6)
    print("ok test_scripted_roundtrip")


def main():
    test_token_feats()
    net = Big2NetSeqAZ(load_token_feats()).eval()
    with torch.no_grad():
        test_causal(net)
        test_kv_equivalence(net, fp16_pool=False)
        test_kv_equivalence(net, fp16_pool=True)
        test_bos_alignment(net)
        test_scripted_roundtrip(net)
    assert SEQ_CAP >= 66
    print("All seq-model tests passed.")


if __name__ == "__main__":
    main()
