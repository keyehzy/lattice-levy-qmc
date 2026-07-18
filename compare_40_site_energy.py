from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from interacting_lattice_levy import (
    ContinuousConfiguration,
    ContinuousPath,
    InteractingLatticeLevySampler,
    _OccupancyIndex,
    pair_overlap_time,
)


@dataclass(frozen=True)
class ReferencePoint:
    U: float
    rdmft1: float
    rdmft2: float
    qmc: float


REFERENCE = (
    ReferencePoint(0.0, -80.0, -80.0, -80.0),
    ReferencePoint(1.2, -57.0, -61.0, -63.0),
    ReferencePoint(2.3, -41.5, -48.0, -49.5),
    ReferencePoint(3.7, -32.5, -38.0, -39.5),
    ReferencePoint(5.0, -25.0, -30.0, -31.5),
    ReferencePoint(6.2, -20.5, -26.5, -27.5),
    ReferencePoint(7.5, -17.5, -23.0, -23.5),
    ReferencePoint(8.7, -15.0, -19.5, -20.5),
    ReferencePoint(10.0, -13.0, -16.5, -17.5),
    ReferencePoint(11.2, -11.8, -14.5, -15.3),
    ReferencePoint(12.4, -10.8, -13.0, -13.8),
    ReferencePoint(13.6, -10.0, -11.8, -12.5),
    ReferencePoint(14.8, -9.5, -10.8, -11.4),
    ReferencePoint(16.1, -9.0, -10.0, -10.5),
    ReferencePoint(17.3, -8.7, -9.5, -9.8),
    ReferencePoint(18.7, -8.3, -9.0, -9.2),
    ReferencePoint(20.0, -8.0, -8.0, -8.0),
)


@dataclass(frozen=True)
class RunTask:
    U: float
    beta: float
    slab_duration: float
    burn_in: int
    samples: int
    batch_size: int
    seed: int
    initialization: str
    phase: str
    N: int = 40
    L: int = 40
    t: float = 1.0


@dataclass
class RunResult:
    U: float
    beta: float
    slab_duration: float
    burn_in: int
    samples: int
    batch_size: int
    seed: int
    initialization: str
    phase: str
    total_energy: float
    total_error: float
    kinetic_energy: float
    interaction_energy: float
    pair_count: float
    acceptance: float
    topology_rate: float
    energy_ess: float
    elapsed_seconds: float
    first_half: float
    second_half: float
    drift_z: float
    cache_error: float


def reference_for(U: float) -> ReferencePoint:
    for point in REFERENCE:
        if math.isclose(U, point.U, rel_tol=0.0, abs_tol=1e-12):
            return point
    available = ", ".join(f"{point.U:g}" for point in REFERENCE)
    raise ValueError(f"U={U:g} is not in the supplied table; choose from {available}")


def calibrated_slab_duration(U: float, t: float) -> float:
    r"""Return the manually calibrated physical slab duration.

    The 40-site experiment used a duration of 1/t through U/t=5 and 5/U
    above it.  The sampler receives ``fraction=duration/beta``.
    """
    if U <= 0.0:
        raise ValueError("The interacting slab schedule requires U > 0")
    return 1.0 / t if U / t <= 5.0 else 5.0 / U


def strong_coupling_energy(U: float, L: int = 40, t: float = 1.0) -> float:
    r"""Fourteenth-order unit-filling 1D ground-state energy.

    This is Eq. (66) of Ejima et al., Phys. Rev. A 85, 053644 (2012),
    using the coefficients extended by Damski and Zakrzewski.  It is used only
    in the strong-coupling comparison, conventionally U/t >= 5 here.
    """
    if U <= 0.0:
        raise ValueError("U must be positive")
    x = t / U
    scaled = (
        -(x**2)
        + x**4
        + (68.0 / 9.0) * x**6
        - (1267.0 / 81.0) * x**8
        + (44171.0 / 1458.0) * x**10
        - (4902596.0 / 6561.0) * x**12
        - (8020902135607.0 / 2645395200.0) * x**14
    )
    return float(4.0 * U * L * scaled)


def effective_sample_size(values: np.ndarray) -> float:
    """Estimate ESS using Geyer's initial-positive-sequence truncation."""
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


def install_unit_filled_mott_state(
    sampler: InteractingLatticeLevySampler,
) -> None:
    """Replace the ideal initialization by one static boson per site."""
    if sampler.d != 1 or sampler.N != sampler.L:
        raise ValueError("Mott initialization requires d=1 and N=L")

    paths: list[ContinuousPath] = []
    for site in range(sampler.L):
        position = np.array([site], dtype=np.int64)
        paths.append(
            ContinuousPath(
                duration=sampler.beta,
                start=position,
                event_times=np.empty(0, dtype=np.float64),
                jumps=np.empty((0, 1), dtype=np.int64),
                end=position,
            )
        )

    state = ContinuousConfiguration(
        L=sampler.L,
        d=1,
        N=sampler.N,
        beta=sampler.beta,
        t=sampler.t,
        cycles=[(label,) for label in range(sampler.N)],
        permutation=np.arange(sampler.N, dtype=np.int64),
        worldlines=paths,
        log_Z0_N=sampler.state.log_Z0_N,
    )
    state.validate()
    sampler.state = state
    sampler.pair_overlap = 0.0
    sampler.action = 0.0
    # The library currently has no public state-installation method, so the
    # experiment must rebuild its private accepted-state occupancy cache.
    sampler._occupancy_index = _OccupancyIndex.from_state(state)


def half_drift_z(batch_means: np.ndarray) -> tuple[float, float, float]:
    midpoint = batch_means.size // 2
    first = batch_means[:midpoint]
    second = batch_means[midpoint:]
    first_mean = float(np.mean(first))
    second_mean = float(np.mean(second))
    first_error = float(np.std(first, ddof=1) / np.sqrt(first.size))
    second_error = float(np.std(second, ddof=1) / np.sqrt(second.size))
    difference_error = math.hypot(first_error, second_error)
    if difference_error == 0.0:
        z_score = 0.0 if first_mean == second_mean else math.inf
    else:
        z_score = (second_mean - first_mean) / difference_error
    return first_mean, second_mean, float(z_score)


def run_task(task: RunTask) -> RunResult:
    if task.slab_duration <= 0.0 or task.slab_duration > task.beta:
        raise ValueError("slab duration must lie in (0, beta]")

    sampler = InteractingLatticeLevySampler(
        N=task.N,
        L=task.L,
        beta=task.beta,
        d=1,
        t=task.t,
        U=task.U,
        rng=np.random.default_rng(task.seed),
    )
    if task.initialization == "mott":
        install_unit_filled_mott_state(sampler)
    elif task.initialization != "ideal":
        raise ValueError(f"unknown initialization {task.initialization!r}")

    fraction = task.slab_duration / task.beta
    for _ in range(task.burn_in):
        sampler.random_seam_stitch_sweep(
            updates=task.N,
            fraction=fraction,
        )

    statistics = sampler.statistics["stitch"]
    attempts_before = statistics.attempts
    accepts_before = statistics.accepts
    topology_before = statistics.topology_changes

    total = np.empty(task.samples)
    kinetic = np.empty(task.samples)
    interaction = np.empty(task.samples)
    pair_count = np.empty(task.samples)
    started = time.perf_counter()
    for sample in range(task.samples):
        sampler.random_seam_stitch_sweep(
            updates=task.N,
            fraction=fraction,
        )
        observable = sampler.observables()
        total[sample] = float(observable["total_energy"])
        kinetic[sample] = float(observable["kinetic_energy"])
        interaction[sample] = float(observable["interaction_energy"])
        pair_count[sample] = (
            float(observable["pair_overlap_time"]) / task.beta
        )
    elapsed = time.perf_counter() - started

    batch_means = total.reshape(-1, task.batch_size).mean(axis=1)
    mean = float(np.mean(batch_means))
    error = float(np.std(batch_means, ddof=1) / np.sqrt(batch_means.size))
    first_half, second_half, drift_z = half_drift_z(batch_means)
    attempts = statistics.attempts - attempts_before
    exact_overlap = pair_overlap_time(sampler.state)

    return RunResult(
        U=task.U,
        beta=task.beta,
        slab_duration=task.slab_duration,
        burn_in=task.burn_in,
        samples=task.samples,
        batch_size=task.batch_size,
        seed=task.seed,
        initialization=task.initialization,
        phase=task.phase,
        total_energy=mean,
        total_error=error,
        kinetic_energy=float(np.mean(kinetic)),
        interaction_energy=float(np.mean(interaction)),
        pair_count=float(np.mean(pair_count)),
        acceptance=(statistics.accepts - accepts_before) / attempts,
        topology_rate=(statistics.topology_changes - topology_before) / attempts,
        energy_ess=effective_sample_size(total),
        elapsed_seconds=elapsed,
        first_half=first_half,
        second_half=second_half,
        drift_z=drift_z,
        cache_error=float(sampler.pair_overlap - exact_overlap),
    )


def analytic_u_zero(t: float) -> RunResult:
    energy = -2.0 * t * 40
    return RunResult(
        U=0.0,
        beta=math.inf,
        slab_duration=0.0,
        burn_in=0,
        samples=0,
        batch_size=0,
        seed=0,
        initialization="analytic",
        phase="ground_limit",
        total_energy=energy,
        total_error=0.0,
        kinetic_energy=energy,
        interaction_energy=0.0,
        pair_count=19.5,
        acceptance=math.nan,
        topology_rate=math.nan,
        energy_ess=math.inf,
        elapsed_seconds=0.0,
        first_half=energy,
        second_half=energy,
        drift_z=0.0,
        cache_error=0.0,
    )


def execute(tasks: list[RunTask], workers: int) -> list[RunResult]:
    if not tasks:
        return []
    results: list[RunResult] = []
    if workers == 1:
        for task in tasks:
            result = run_task(task)
            results.append(result)
            print_progress(result)
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print_progress(result)
    return results


def print_progress(result: RunResult) -> None:
    print(
        f"finished {result.phase:4s}: U={result.U:4.1f}, beta={result.beta:g}, "
        f"E={result.total_energy:.4f} +/- {result.total_error:.4f}, "
        f"accept={result.acceptance:.3f}, drift_z={result.drift_z:+.2f}",
        flush=True,
    )


def metric(values: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    difference = values - target
    return (
        float(np.mean(np.abs(difference))),
        float(np.sqrt(np.mean(difference**2))),
        float(np.mean(difference)),
    )


def chosen_results(
    main: list[RunResult],
    spots: list[RunResult],
) -> dict[float, RunResult]:
    chosen = {result.U: result for result in main}
    for result in spots:
        chosen[result.U] = result
    return chosen


def print_summary(
    selected_points: list[ReferencePoint],
    main: list[RunResult],
    spots: list[RunResult],
    t: float,
) -> None:
    chosen = chosen_results(main, spots)
    if any(point.U == 0.0 for point in selected_points):
        chosen[0.0] = analytic_u_zero(t)

    print("\nChosen comparison (spot runs replace main runs at the same U)")
    print(
        " U/t    supplied QMC       code +/- batch SE      delta  beta"
        "   Tslab   accept  drift-z"
    )
    for point in selected_points:
        result = chosen[point.U]
        beta = "inf" if math.isinf(result.beta) else f"{result.beta:g}"
        acceptance = (
            "   -  " if math.isnan(result.acceptance) else f"{result.acceptance:6.3f}"
        )
        print(
            f"{point.U:5.1f} {point.qmc:14.3f}"
            f" {result.total_energy:11.3f} +/- {result.total_error:7.3f}"
            f" {result.total_energy - point.qmc:+9.3f} {beta:>5s}"
            f" {result.slab_duration:8.3f} {acceptance} {result.drift_z:+8.2f}"
        )

    code = np.array([chosen[point.U].total_energy for point in selected_points])
    for name, target in (
        ("RDMFT1", np.array([point.rdmft1 for point in selected_points])),
        ("RDMFT2", np.array([point.rdmft2 for point in selected_points])),
        ("supplied QMC", np.array([point.qmc for point in selected_points])),
    ):
        mae, rmse, bias = metric(code, target)
        print(f"{name:12s}: MAE={mae:.4f}, RMSE={rmse:.4f}, bias={bias:+.4f}")

    strong_points = [point for point in selected_points if point.U / t >= 5.0]
    if strong_points:
        strong_code = np.array(
            [chosen[point.U].total_energy for point in strong_points]
        )
        strong_target = np.array(
            [strong_coupling_energy(point.U, t=t) for point in strong_points]
        )
        mae, rmse, bias = metric(strong_code, strong_target)
        print(
            "strong series: "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}, bias={bias:+.4f}"
        )

    if spots:
        main_by_u = {result.U: result for result in main}
        print("\nGround-projection spot checks")
        for spot in sorted(spots, key=lambda result: result.U):
            base = main_by_u.get(spot.U)
            if base is None:
                continue
            print(
                f"U={spot.U:g}: beta={base.beta:g} E={base.total_energy:.4f}"
                f" -> beta={spot.beta:g} E={spot.total_energy:.4f}"
            )


def write_csv(
    path: Path,
    results: list[RunResult],
    t: float,
) -> None:
    rows: list[dict[str, object]] = []
    for result in sorted(results, key=lambda item: (item.U, item.beta)):
        row: dict[str, object] = asdict(result)
        reference = reference_for(result.U)
        row.update(
            rdmft1=reference.rdmft1,
            rdmft2=reference.rdmft2,
            supplied_qmc=reference.qmc,
            delta_qmc=result.total_energy - reference.qmc,
            strong_coupling=(
                strong_coupling_energy(result.U, t=t)
                if result.U / t >= 5.0
                else math.nan
            ),
        )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def write_plot(
    path: Path,
    selected_points: list[ReferencePoint],
    main: list[RunResult],
    spots: list[RunResult],
    t: float,
) -> None:
    import matplotlib.pyplot as plt

    chosen = chosen_results(main, spots)
    if any(point.U == 0.0 for point in selected_points):
        chosen[0.0] = analytic_u_zero(t)
    U = np.array([point.U for point in selected_points])

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(U, [point.rdmft1 for point in selected_points], "o-", label="RDMFT1")
    axis.plot(U, [point.rdmft2 for point in selected_points], "o-", label="RDMFT2")
    axis.plot(U, [point.qmc for point in selected_points], "o-", label="supplied QMC")
    axis.errorbar(
        U,
        [chosen[point.U].total_energy for point in selected_points],
        yerr=[chosen[point.U].total_error for point in selected_points],
        fmt="s-",
        capsize=3,
        label="this code",
    )
    strong_u = U[U / t >= 5.0]
    if strong_u.size:
        axis.plot(
            strong_u,
            [strong_coupling_energy(value, t=t) for value in strong_u],
            "--",
            label="14th-order strong coupling",
        )
    axis.set_xlabel("U/t")
    axis.set_ylabel("total energy")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"wrote {path}")


def validate_batching(samples: int, batch_size: int, name: str) -> None:
    if samples < 4 * batch_size or samples % batch_size:
        raise ValueError(
            f"{name}: samples must be a multiple of batch size and provide "
            "at least four batches"
        )
    if (samples // batch_size) % 2:
        raise ValueError(f"{name}: the number of batches must be even")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the N=L=40 Bose-Hubbard energy comparison."
    )
    parser.add_argument(
        "--U-values",
        type=float,
        nargs="+",
        default=[point.U for point in REFERENCE],
    )
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--initialization", choices=("mott", "ideal"), default="mott")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--spot-U", type=float, nargs="*", default=[8.7])
    parser.add_argument("--spot-beta", type=float, default=40.0)
    parser.add_argument("--spot-burn-in", type=int, default=900)
    parser.add_argument("--spot-samples", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short workflow test; its energies are not production results.",
    )
    args = parser.parse_args()

    if args.beta <= 0.0 or args.spot_beta <= 0.0 or args.t <= 0.0:
        raise ValueError("beta values and t must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.burn_in is not None and args.burn_in < 0:
        raise ValueError("burn-in must be nonnegative")
    if args.spot_burn_in < 0:
        raise ValueError("spot burn-in must be nonnegative")
    selected_points = sorted(
        {reference_for(U).U: reference_for(U) for U in args.U_values}.values(),
        key=lambda point: point.U,
    )

    if args.quick:
        print("QUICK MODE: outputs test the workflow and are not converged.")
        main_burn_in = 20
        main_samples = 40
        batch_size = 10
        spot_burn_in = 20
        spot_samples = 40
    else:
        main_burn_in = args.burn_in
        main_samples = args.samples
        batch_size = args.batch_size
        spot_burn_in = args.spot_burn_in
        spot_samples = args.spot_samples

    tasks: list[RunTask] = []
    for index, point in enumerate(selected_points):
        if point.U == 0.0:
            continue
        weak = point.U / args.t <= 5.0
        burn_in = main_burn_in if main_burn_in is not None else (500 if weak else 700)
        samples = (
            main_samples
            if main_samples is not None
            else (1_000 if weak else 1_200)
        )
        validate_batching(samples, batch_size, f"U={point.U:g}")
        tasks.append(
            RunTask(
                U=point.U,
                beta=args.beta,
                slab_duration=min(
                    args.beta,
                    calibrated_slab_duration(point.U, args.t),
                ),
                burn_in=burn_in,
                samples=samples,
                batch_size=batch_size,
                seed=args.seed + index,
                initialization=args.initialization,
                phase="main",
                t=args.t,
            )
        )

    spot_tasks: list[RunTask] = []
    selected_u = {point.U for point in selected_points}
    for index, requested_u in enumerate(args.spot_U):
        point = reference_for(requested_u)
        if point.U == 0.0 or point.U not in selected_u:
            continue
        validate_batching(spot_samples, batch_size, f"spot U={point.U:g}")
        spot_tasks.append(
            RunTask(
                U=point.U,
                beta=args.spot_beta,
                slab_duration=min(
                    args.spot_beta,
                    calibrated_slab_duration(point.U, args.t),
                ),
                burn_in=spot_burn_in,
                samples=spot_samples,
                batch_size=batch_size,
                seed=args.seed + 10_000 + index,
                initialization=args.initialization,
                phase="spot",
                t=args.t,
            )
        )

    print(
        f"N=L=40, t={args.t:g}, beta={args.beta:g}, "
        f"initialization={args.initialization}, workers={args.workers}"
    )
    print("slab schedule: T=1/t for U/t<=5, otherwise T=5/U")
    started = time.perf_counter()
    main_results = execute(tasks, args.workers)
    spot_results = execute(spot_tasks, min(args.workers, len(spot_tasks) or 1))
    elapsed = time.perf_counter() - started

    print_summary(selected_points, main_results, spot_results, args.t)
    print(f"\ntotal wall time: {elapsed:.1f} s")

    all_results = main_results + spot_results
    if any(point.U == 0.0 for point in selected_points):
        all_results.append(analytic_u_zero(args.t))
    if args.output and all_results:
        write_csv(args.output, all_results, args.t)
    if args.plot:
        write_plot(args.plot, selected_points, main_results, spot_results, args.t)


if __name__ == "__main__":
    main()
