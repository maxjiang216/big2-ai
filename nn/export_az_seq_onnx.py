"""Export the scripted az_seq champion to ONNX for the browser (onnxruntime-web).

    uv run python -m nn.export_az_seq_onnx --model models/az_seq.pt \\
        --out web/model.onnx

The exported graph is a single full-recompute leaf eval (no KV cache — the JS
MCTS recomputes the trunk per leaf, corintho-style):

    inputs:
      tokens [1, T] int64   move-id history ([move_0 .. move_{T-1}]; BOS added
                            internally by forward_full)
      hand   [1, 48] f32    owner exact card features
      oppmax [1, 48] f32    opponent thermo
      sides  [1, 5]  f32    [osz, usz, otm, mpts, opts]
    outputs:
      value    [1, 1]
      policy   [1, 138]
      behavior [1, 458]
      qa       [1, 458]

The decision hidden is read at sequence index T (= len(tokens)): the trunk
output is [1, T+1, d] (BOS prepended), so index T is the last row.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from nn.model_az_seq import Big2NetSeqAZ, load_token_feats


class OnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.m = model

    def forward(self, tokens, hand, oppmax, sides):
        h = self.m.forward_full(tokens)  # [1, T+1, d]
        hidden = h[:, -1, :]  # decision at index T (last row)
        osz = sides[:, 0]
        usz = sides[:, 1]
        otm = sides[:, 2]
        mpts = sides[:, 3]
        opts = sides[:, 4]
        return self.m.readout(hidden, hand, oppmax, osz, usz, otm, mpts, opts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/az_seq.pt")
    p.add_argument("--out", default="web/model.onnx")
    p.add_argument("--opset", type=int, default=17)
    cfg = p.parse_args()

    scripted = torch.jit.load(cfg.model, map_location="cpu")
    scripted.eval()
    n_layers, n_heads, _head_dim, d_model, seq_cap = scripted.config()
    sd = scripted.state_dict()
    d_ff = sd["blocks.0.ff1.weight"].shape[0]
    model = Big2NetSeqAZ(
        load_token_feats(),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        seq_cap=seq_cap,
    )
    # Scripted surface drops training-only heads (margin / opp_hand); the
    # exported graph only uses value/policy/behavior/qa, so strict=False is safe.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all(
        "margin_head" in k or "opp_hand_head" in k for k in missing
    ), f"unexpected missing keys: {missing}"
    model.eval()
    wrapper = OnnxWrapper(model).eval()

    T = 7
    tokens = torch.randint(0, 100, (1, T), dtype=torch.int64)
    hand = torch.randn(1, 48)
    oppmax = torch.randn(1, 48)
    sides = torch.rand(1, 5)
    args = (tokens, hand, oppmax, sides)

    with torch.no_grad():
        ref = wrapper(*args)

    torch.onnx.export(
        wrapper,
        args,
        cfg.out,
        input_names=["tokens", "hand", "oppmax", "sides"],
        output_names=["value", "policy", "behavior", "qa"],
        dynamic_axes={"tokens": {1: "T"}},
        opset_version=cfg.opset,
        dynamo=False,
    )
    print(f"✓ exported {cfg.model} -> {cfg.out}")

    # Parity: onnxruntime vs pytorch, on the export sample AND a different T.
    import onnxruntime as ort

    sess = ort.InferenceSession(cfg.out, providers=["CPUExecutionProvider"])
    names = ["value", "policy", "behavior", "qa"]

    def check(toks, hd, op, sd, tag):
        feeds = {
            "tokens": toks.numpy(),
            "hand": hd.numpy(),
            "oppmax": op.numpy(),
            "sides": sd.numpy(),
        }
        outs = sess.run(names, feeds)
        with torch.no_grad():
            r = wrapper(toks, hd, op, sd)
        ok = True
        for i, nm in enumerate(names):
            d = float(np.abs(outs[i] - r[i].numpy()).max())
            flag = "ok" if d < 1e-3 else "FAIL"
            if d >= 1e-3:
                ok = False
            print(f"  [{tag}] {nm:9s} max|Δ| = {d:.2e}  {flag}")
        return ok

    ok1 = check(tokens, hand, oppmax, sides, f"T={T}")
    T2 = 15
    ok2 = check(
        torch.randint(0, 100, (1, T2), dtype=torch.int64),
        torch.randn(1, 48),
        torch.randn(1, 48),
        torch.rand(1, 5),
        f"T={T2}",
    )
    print("PARITY PASS" if (ok1 and ok2) else "PARITY FAIL")


if __name__ == "__main__":
    main()
