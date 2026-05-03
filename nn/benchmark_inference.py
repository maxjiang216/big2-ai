"""Benchmark Big2Net inference throughput.

Usage:
    uv run python -m nn.benchmark_inference
    uv run python -m nn.benchmark_inference --device cpu
    uv run python -m nn.benchmark_inference --warmup 50 --reps 200
"""

from __future__ import annotations

import argparse
import time

import torch

from nn.model import Big2Net, ENCODING_DIM

# Measured from 10k-game sample.
AVG_TURNS_PER_GAME = 17.2
# Mean legal moves per decision point (from 50k-game measurement).
MEAN_K_LEAD = 11.6
MEAN_K_RESP = 3.85
# Mean NN calls per game if scoring every legal move.
MEAN_NN_CALLS_PER_GAME = 150.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _median_ms(times: list[float]) -> float:
    times.sort()
    return times[len(times) // 2] * 1000


def bench_full_forward(
    model: Big2Net,
    device: torch.device,
    batch_sizes: list[int],
    warmup: int,
    reps: int,
) -> None:
    """Baseline: one forward() call per position (opp duplicated in batch)."""
    print(f"\n--- full forward() ---")
    print(f"{'batch':>8}  {'ms':>8}  {'pos/s':>12}  {'games/s (÷150)':>16}")
    print("-" * 52)

    for bs in batch_sizes:
        hand = torch.rand(bs, ENCODING_DIM, device=device)
        opp = torch.rand(bs, ENCODING_DIM, device=device)
        move = torch.rand(bs, ENCODING_DIM, device=device)
        hint = torch.rand(bs, device=device)
        flag = torch.zeros(bs, dtype=torch.bool, device=device)
        pass_ = torch.zeros(bs, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(warmup):
                model(hand, opp, move, hint, flag, pass_)
        _sync(device)

        times = []
        with torch.no_grad():
            for _ in range(reps):
                t0 = time.perf_counter()
                model(hand, opp, move, hint, flag, pass_)
                _sync(device)
                times.append(time.perf_counter() - t0)

        ms = _median_ms(times)
        pos_s = bs / (ms / 1000)
        print(
            f"{bs:>8}  {ms:>8.3f}  {pos_s:>12,.0f}  {pos_s/MEAN_NN_CALLS_PER_GAME:>16,.0f}"
        )
    print()


def bench_amortised(
    model: Big2Net,
    device: torch.device,
    G: int,
    K_values: list[float],
    warmup: int,
    reps: int,
) -> None:
    """Amortised: encode_opp() once per game, forward_with_opp() on G*K batch.

    Simulates the C++ self-play loop pattern:
      - G games, each at a decision point with K legal moves
      - encode_opp called once per game (G calls)
      - forward_with_opp called once with the G*K batch
    """
    print(f"\n--- amortised (G={G} parallel games) ---")
    print(
        f"{'K':>6}  {'ms':>8}  {'pos/s':>12}  {'games/s':>10}  {'speedup vs full':>16}"
    )
    print("-" * 58)

    # Baseline time for G*K positions with full forward (for speedup calc)
    def full_time(GK: int) -> float:
        hand = torch.rand(GK, ENCODING_DIM, device=device)
        opp = torch.rand(GK, ENCODING_DIM, device=device)
        move = torch.rand(GK, ENCODING_DIM, device=device)
        hint = torch.rand(GK, device=device)
        flag = torch.zeros(GK, dtype=torch.bool, device=device)
        pass_ = torch.zeros(GK, dtype=torch.bool, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                model(hand, opp, move, hint, flag, pass_)
        _sync(device)
        times = []
        with torch.no_grad():
            for _ in range(reps):
                t0 = time.perf_counter()
                model(hand, opp, move, hint, flag, pass_)
                _sync(device)
                times.append(time.perf_counter() - t0)
        return _median_ms(times)

    for K in K_values:
        Ki = round(K)
        GK = G * Ki
        baseline_ms = full_time(GK)

        # Amortised inputs
        opp_g = torch.rand(G, ENCODING_DIM, device=device)
        hand_gk = torch.rand(GK, ENCODING_DIM, device=device)
        move_gk = torch.rand(GK, ENCODING_DIM, device=device)
        hint_gk = torch.rand(GK, device=device)
        flag_gk = torch.zeros(GK, dtype=torch.bool, device=device)
        pass_gk = torch.zeros(GK, dtype=torch.bool, device=device)

        # encode_opp on G games, tile each Ki times → [G*Ki, 96]
        with torch.no_grad():
            for _ in range(warmup):
                o_g = model.encode_opp(opp_g)
                o_gk = o_g.repeat_interleave(Ki, dim=0)
                model.forward_with_opp(
                    hand_gk, move_gk, hint_gk, flag_gk, pass_gk, o_gk
                )
        _sync(device)

        times = []
        with torch.no_grad():
            for _ in range(reps):
                t0 = time.perf_counter()
                o_g = model.encode_opp(opp_g)
                o_gk = o_g.repeat_interleave(Ki, dim=0)
                model.forward_with_opp(
                    hand_gk, move_gk, hint_gk, flag_gk, pass_gk, o_gk
                )
                _sync(device)
                times.append(time.perf_counter() - t0)

        ms = _median_ms(times)
        pos_s = GK / (ms / 1000)
        game_s = pos_s / (K * AVG_TURNS_PER_GAME / AVG_TURNS_PER_GAME * K)
        # games/s = positions/s ÷ K (moves per decision) ÷ (17 decisions/game)
        game_s = pos_s / (K * AVG_TURNS_PER_GAME)
        speedup = baseline_ms / ms
        label = f"K={K:.1f}"
        print(
            f"{label:>6}  {ms:>8.3f}  {pos_s:>12,.0f}  {game_s:>10,.0f}  {speedup:>15.2f}x"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument(
        "--games",
        type=int,
        default=512,
        help="Parallel games G for amortised benchmark",
    )
    parser.add_argument("--batch-sizes", default="1,32,128,512,2048")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = Big2Net().to(device).eval()
    print(
        f"Device: {device}  |  params: {sum(p.numel() for p in model.parameters()):,}"
    )

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    bench_full_forward(model, device, batch_sizes, args.warmup, args.reps)
    bench_amortised(
        model, device, args.games, [MEAN_K_RESP, MEAN_K_LEAD], args.warmup, args.reps
    )


if __name__ == "__main__":
    main()
