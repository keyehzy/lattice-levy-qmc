from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from interacting_lattice_levy import InteractingLatticeLevySampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=2.0)
    parser.add_argument("--stitch-fraction", type=float, default=0.75)
    parser.add_argument("--sweeps", type=int, default=3000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--output", type=Path, default=Path("interacting_trace.png")
    )
    args = parser.parse_args()

    sampler = InteractingLatticeLevySampler(
        N=args.N,
        beta=args.beta,
        L=args.L,
        d=1,
        t=args.t,
        U=args.U,
        rng=np.random.default_rng(args.seed),
    )

    for _ in range(args.burn_in):
        sampler.random_seam_stitch_sweep(
            updates=max(1, args.N // 2),
            fraction=args.stitch_fraction,
        )

    energy = np.empty(args.sweeps)
    double_occupancy = np.empty(args.sweeps)
    winding_squared = np.empty(args.sweeps)

    for sweep in range(args.sweeps):
        sampler.random_seam_stitch_sweep(
            updates=max(1, args.N // 2),
            fraction=args.stitch_fraction,
        )
        observable = sampler.observables()
        energy[sweep] = float(observable["total_energy"])
        double_occupancy[sweep] = float(observable["double_occupancy_per_site"])
        winding = np.asarray(observable["winding"])
        winding_squared[sweep] = float(winding @ winding)

    running_energy = np.cumsum(energy) / np.arange(1, args.sweeps + 1)
    running_double = np.cumsum(double_occupancy) / np.arange(1, args.sweeps + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(running_energy)
    axes[0].set_ylabel("running total energy")
    axes[1].plot(running_double)
    axes[1].set_ylabel("running pair occupancy/site")
    axes[1].set_xlabel("measurement sweep")
    fig.suptitle(
        f"Continuous-time lattice-Levy Bose-Hubbard: "
        f"N={args.N}, L={args.L}, beta={args.beta}, U/t={args.U/args.t:g}"
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)

    print(f"saved {args.output}")
    print(f"<E> = {energy.mean():.8f}")
    print(f"<D>/site = {double_occupancy.mean():.8f}")
    print(f"<W^2> = {winding_squared.mean():.8f}")
    for name, stats in sampler.statistics.items():
        print(f"{name} acceptance = {stats.acceptance:.6f}")
        if stats.topology_changes:
            print(
                f"{name} topology changes/attempt = "
                f"{stats.topology_change_rate:.6f}"
            )


if __name__ == "__main__":
    main()
