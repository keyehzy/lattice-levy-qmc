
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lattice_levy import sample_ideal_boson_configuration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("boson_worldlines_demo.png"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = sample_ideal_boson_configuration(
        N=args.N,
        beta=args.beta,
        M=args.M,
        L=args.L,
        d=1,
        t=args.t,
        rng=rng,
    )

    tau = np.linspace(0.0, args.beta, args.M + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for particle in range(args.N):
        ax.step(
            tau,
            cfg.worldlines[particle, :, 0],
            where="post",
            label=f"{particle} -> {cfg.permutation[particle]}",
        )

    cycle_text = ", ".join(str(c.labels) for c in cfg.cycles)
    ax.set(
        xlabel=r"$\tau$",
        ylabel="site (mod L)",
        title=f"Ideal lattice bosons; cycles: {cycle_text}",
        yticks=np.arange(args.L),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")
    print(f"permutation = {cfg.permutation.tolist()}")
    print(f"log Z_N = {cfg.log_ZN:.12g}")
    for idx, cycle in enumerate(cfg.cycles):
        print(
            f"cycle {idx}: labels={cycle.labels}, "
            f"base={cycle.base_point.tolist()}, "
            f"winding={cycle.winding.tolist()}"
        )


if __name__ == "__main__":
    main()
