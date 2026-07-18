from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from interacting_lattice_levy import InteractingLatticeLevySampler


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first, *rest)


def exact_diagonalization(
    N: int,
    L: int,
    beta: float,
    t: float,
    U: float,
) -> dict[str, float]:
    if L < 3:
        raise ValueError("This compact validator assumes a 1D ring with L >= 3.")

    basis = list(compositions(N, L))
    index = {state: i for i, state in enumerate(basis)}
    dimension = len(basis)
    hamiltonian = np.zeros((dimension, dimension), dtype=np.float64)
    pair_operator = np.zeros(dimension, dtype=np.float64)

    for column, occupation_tuple in enumerate(basis):
        occupation = list(occupation_tuple)
        pair_operator[column] = sum(n * (n - 1) / 2 for n in occupation)
        hamiltonian[column, column] = U * pair_operator[column]

        for site in range(L):
            neighbor = (site + 1) % L
            # a_site^dagger a_neighbor
            if occupation[neighbor] > 0:
                final = occupation.copy()
                final[neighbor] -= 1
                final[site] += 1
                row = index[tuple(final)]
                hamiltonian[row, column] += -t * math.sqrt(
                    (occupation[site] + 1) * occupation[neighbor]
                )
            # a_neighbor^dagger a_site
            if occupation[site] > 0:
                final = occupation.copy()
                final[site] -= 1
                final[neighbor] += 1
                row = index[tuple(final)]
                hamiltonian[row, column] += -t * math.sqrt(
                    (occupation[neighbor] + 1) * occupation[site]
                )

    eigenvalues, eigenvectors = eigh(hamiltonian)
    shifted_weights = np.exp(-beta * (eigenvalues - eigenvalues.min()))
    normalization = float(shifted_weights.sum())
    pair_in_eigenstates = (np.abs(eigenvectors) ** 2).T @ pair_operator

    total_energy = float(shifted_weights @ eigenvalues / normalization)
    pair_count = float(shifted_weights @ pair_in_eigenstates / normalization)
    interaction_energy = U * pair_count
    kinetic_energy = total_energy - interaction_energy

    return {
        "total_energy": total_energy,
        "kinetic_energy": kinetic_energy,
        "interaction_energy": interaction_energy,
        "pair_count": pair_count,
        "hilbert_dimension": float(dimension),
    }


def batch_mean_and_error(values: np.ndarray, batch_size: int) -> tuple[float, float]:
    n_batches = values.size // batch_size
    if n_batches < 2:
        raise ValueError("Need at least two complete batches")
    trimmed = values[: n_batches * batch_size]
    batch_means = trimmed.reshape(n_batches, batch_size).mean(axis=1)
    return float(batch_means.mean()), float(
        batch_means.std(ddof=1) / np.sqrt(n_batches)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path, default=Path("interacting_ed_validation.txt"))
    args = parser.parse_args()

    N = 2
    L = 3
    beta = 0.8
    t = 1.0
    U = 1.2

    exact = exact_diagonalization(N=N, L=L, beta=beta, t=t, U=U)
    sampler = InteractingLatticeLevySampler(
        N=N,
        beta=beta,
        L=L,
        d=1,
        t=t,
        U=U,
        rng=np.random.default_rng(args.seed),
    )

    # A pure global ideal-gas independence proposal is deliberately used here:
    # the comparison tests the finite-U reweighting without relying on local
    # update tuning.
    for _ in range(args.burn_in):
        sampler.global_update()

    total_energy = np.empty(args.samples)
    kinetic_energy = np.empty(args.samples)
    interaction_energy = np.empty(args.samples)
    pair_count = np.empty(args.samples)

    for sample in range(args.samples):
        sampler.global_update()
        observable = sampler.observables()
        total_energy[sample] = float(observable["total_energy"])
        kinetic_energy[sample] = float(observable["kinetic_energy"])
        interaction_energy[sample] = float(observable["interaction_energy"])
        pair_count[sample] = float(observable["pair_overlap_time"]) / beta

    measured = {}
    for name, values in (
        ("total_energy", total_energy),
        ("kinetic_energy", kinetic_energy),
        ("interaction_energy", interaction_energy),
        ("pair_count", pair_count),
    ):
        measured[name] = batch_mean_and_error(values, args.batch_size)

    lines = [
        "Finite-U continuous-time lattice-Levy validation",
        "=================================================",
        f"N={N}, L={L}, d=1, beta={beta}, t={t}, U={U}",
        f"samples={args.samples}, burn_in={args.burn_in}, batch_size={args.batch_size}",
        f"global acceptance={sampler.statistics['global'].acceptance:.6f}",
        f"Hilbert dimension={int(exact['hilbert_dimension'])}",
        "",
        "observable              exact                Monte Carlo",
        "--------------------------------------------------------------",
    ]
    for name in ("total_energy", "kinetic_energy", "interaction_energy", "pair_count"):
        mean, error = measured[name]
        lines.append(
            f"{name:20s} {exact[name]: .10f}    {mean: .10f} +/- {error:.10f}"
        )

    report = "\n".join(lines) + "\n"
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
