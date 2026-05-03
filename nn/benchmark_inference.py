"""Benchmark Big2Net inference throughput.

Usage:
    uv run python nn/benchmark_inference.py
    uv run python nn/benchmark_inference.py --device cpu
    uv run python nn/benchmark_inference.py --warmup 50 --reps 200
"""

from __future__ import annotations

import argparse
import time

import torch

from nn.model import Big2Net, ENCODING_DIM

# Average recorded turns per game (measured from 10k-game sample).
# Used to convert position/s → game-equivalent/s.
AVG_TURNS_PER_GAME = 17.2


def make_batch(batch_size: int, device: torch.device):
    hand  = torch.rand(batch_size, ENCODING_DIM, device=device)
    opp   = torch.rand(batch_size, ENCODING_DIM, device=device)
    move  = torch.rand(batch_size, ENCODING_DIM, device=device)
    hint  = torch.rand(batch_size, device=device)
    flag  = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pass_ = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return hand, opp, move, hint, flag, pass_


def benchmark(
    model: Big2Net,
    device: torch.device,
    batch_sizes: list[int],
    warmup: int,
    reps: int,
) -> None:
    model.eval()
    use_cuda = device.type == "cuda"

    print(f"\nDevice: {device}  |  params: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"{'batch':>8}  {'median ms':>10}  {'positions/s':>13}  {'game-equiv/s':>14}")
    print("-" * 52)

    for bs in batch_sizes:
        inputs = make_batch(bs, device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(*inputs)
        if use_cuda:
            torch.cuda.synchronize()

        # Timed reps
        times = []
        with torch.no_grad():
            for _ in range(reps):
                t0 = time.perf_counter()
                model(*inputs)
                if use_cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        times.sort()
        median_ms = times[len(times) // 2] * 1000
        pos_per_s = bs / (median_ms / 1000)
        game_per_s = pos_per_s / AVG_TURNS_PER_GAME

        print(f"{bs:>8}  {median_ms:>10.3f}  {pos_per_s:>13,.0f}  {game_per_s:>14,.0f}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--batch-sizes", default="1,8,32,128,512,2048")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    model = Big2Net().to(device)
    benchmark(model, device, batch_sizes, args.warmup, args.reps)


if __name__ == "__main__":
    main()
