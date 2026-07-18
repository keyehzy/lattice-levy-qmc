from __future__ import annotations

import argparse
import time

import numpy as np

from interacting_lattice_levy import InteractingLatticeLevySampler
from validate_interacting_ed import batch_mean_and_error, exact_diagonalization


def effective_sample_size(values: np.ndarray) -> float:
    """Estimate ESS with Geyer's initial-positive-sequence truncation."""
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    n = centered.size
    if n < 2:
        return float(n)
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_size)
    autocovariance = np.fft.irfft(spectrum * spectrum.conj(), fft_size)[:n]
    autocovariance /= np.arange(n, 0, -1)
    if autocovariance[0] <= 0.0:
        return float(n)

    correlation = autocovariance / autocovariance[0]
    integrated_time = 1.0
    for lag in range(1, n - 1, 2):
        pair_sum = float(correlation[lag] + correlation[lag + 1])
        if pair_sum <= 0.0:
            break
        integrated_time += 2.0 * pair_sum
    return float(n / max(integrated_time, 1.0))


def run_scaling(args: argparse.Namespace) -> None:
    print("\nFixed-density random-seam scaling (L=N+1)")
    print("N   ms/macro   acceptance   topology/attempt   energy ESS/s")
    for index, N in enumerate(args.sizes):
        sampler = InteractingLatticeLevySampler(
            N=N,
            L=N + 1,
            beta=args.beta,
            d=1,
            t=args.t,
            U=args.scaling_U,
            rng=np.random.default_rng(args.seed + index),
        )
        updates = max(1, N // 2)
        for _ in range(args.burn_in):
            sampler.random_seam_stitch_sweep(
                updates=updates, fraction=args.fraction
            )

        stats = sampler.statistics["stitch"]
        attempts_before = stats.attempts
        accepts_before = stats.accepts
        topology_before = stats.topology_changes
        energy = np.empty(args.samples)
        started = time.perf_counter()
        for sample in range(args.samples):
            sampler.random_seam_stitch_sweep(
                updates=updates, fraction=args.fraction
            )
            energy[sample] = float(sampler.observables()["total_energy"])
        elapsed = time.perf_counter() - started

        attempts = stats.attempts - attempts_before
        acceptance = (stats.accepts - accepts_before) / attempts
        topology_rate = (stats.topology_changes - topology_before) / attempts
        print(
            f"{N:2d}  {1e3 * elapsed / args.samples:9.3f}"
            f"   {acceptance:10.3f}   {topology_rate:16.3f}"
            f"   {effective_sample_size(energy) / elapsed:12.1f}"
        )


def run_interaction_sweep(args: argparse.Namespace) -> None:
    if args.samples < 2 * args.batch_size:
        raise ValueError("samples must contain at least two complete batches")
    print("\nN=2, L=3 comparison with exact diagonalization")
    print("U      exact E       MC E +/- error       z     pair exact/MC   accept")
    for index, U in enumerate(args.U_values):
        exact = exact_diagonalization(2, 3, args.beta, args.t, U)
        sampler = InteractingLatticeLevySampler(
            N=2,
            L=3,
            beta=args.beta,
            d=1,
            t=args.t,
            U=U,
            rng=np.random.default_rng(args.seed + 10_000 + index),
        )
        for _ in range(args.burn_in):
            sampler.random_seam_stitch_sweep(updates=1, fraction=args.fraction)

        energy = np.empty(args.samples)
        pair_count = np.empty(args.samples)
        for sample in range(args.samples):
            sampler.random_seam_stitch_sweep(updates=1, fraction=args.fraction)
            observable = sampler.observables()
            energy[sample] = float(observable["total_energy"])
            pair_count[sample] = (
                float(observable["pair_overlap_time"]) / args.beta
            )

        mean, error = batch_mean_and_error(energy, args.batch_size)
        pair_mean = float(np.mean(pair_count))
        z_score = (mean - exact["total_energy"]) / error
        print(
            f"{U:5g}  {exact['total_energy']:11.6f}"
            f"  {mean:11.6f} +/- {error:8.6f}  {z_score:6.2f}"
            f"   {exact['pair_count']:.5f}/{pair_mean:.5f}"
            f"   {sampler.statistics['stitch'].acceptance:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 6, 12, 24, 32])
    parser.add_argument(
        "--U-values", type=float, nargs="+", default=[0.0, 0.1, 1.2, 10.0, 100.0]
    )
    parser.add_argument("--scaling-U", type=float, default=1.2)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--fraction", type=float, default=0.75)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if args.burn_in < 0 or args.samples < 1 or args.batch_size < 1:
        raise ValueError("run lengths must be positive (burn-in may be zero)")
    run_scaling(args)
    run_interaction_sweep(args)


if __name__ == "__main__":
    main()
