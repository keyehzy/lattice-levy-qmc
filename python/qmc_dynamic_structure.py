"""Batch MaxEnt reconstruction of the dynamic density structure factor.

The C++ continuation bundle stores the connected density response per lattice
site.  For every measured nonzero lattice momentum, this module reconstructs

    A_q(w) = -Im chi^R_nn(q, w) / pi

with the vendored MaxEnt adapter and converts it to the conventional
per-particle positive-frequency dynamic structure factor

    S(q, w) = (V/N) A_q(w) / (1 - exp(-beta*w)).

Block jackknife replicates keep the covariance rotation, retained covariance
modes, real-frequency mesh, default model, alpha mesh, and analyzer settings
fixed.  Only the leave-one-block-out imaginary-axis mean changes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from qmc_maxent import (
    BundleFormatError,
    DensityContinuationData,
    DensityContinuationMomentum,
    MaxEntRun,
    MaxEntRunOptions,
    PreparedMaxEntInput,
    list_density_continuation_momenta,
    load_density_block_values,
    load_density_continuation_bundle,
    prepare_maxent_input,
    run_maxent,
    write_adapter_output,
)


@dataclass(frozen=True)
class MomentumGeometry:
    indices: tuple[int, ...]
    canonical_wavevector: np.ndarray
    centered_wavevector: np.ndarray
    centered_norm: float


@dataclass(frozen=True)
class JackknifeSummary:
    spectra: np.ndarray
    spectral_standard_error: np.ndarray
    static_values: np.ndarray
    static_standard_error: float
    peak_frequencies: np.ndarray
    peak_standard_error: float
    selected_alphas: np.ndarray
    covariance_chi2: np.ndarray
    converged: np.ndarray


@dataclass(frozen=True)
class DynamicStructureResult:
    momentum: DensityContinuationMomentum
    geometry: MomentumGeometry
    data: DensityContinuationData
    prepared: PreparedMaxEntInput
    maxent: MaxEntRun
    structure_factor: np.ndarray
    static_structure_factor: float
    peak_frequency: float
    jackknife: JackknifeSummary | None


def _manifest_nonnegative_int(manifest: Mapping[str, str], key: str) -> int:
    try:
        value = int(manifest[key])
    except (KeyError, ValueError) as error:
        raise BundleFormatError(f"manifest has an invalid integer {key!r}") from error
    if value < 0:
        raise BundleFormatError(f"manifest key {key!r} must be nonnegative")
    return value


def momentum_geometry(
    indices: Sequence[int], linear_size: int
) -> MomentumGeometry:
    """Return canonical and centered first-Brillouin-zone wavevectors."""

    if linear_size <= 0:
        raise ValueError("linear_size must be positive")
    index_array = np.asarray(indices, dtype=int)
    if index_array.ndim != 1 or index_array.size == 0:
        raise ValueError("momentum indices must be a nonempty vector")
    if np.any(index_array < 0) or np.any(index_array >= linear_size):
        raise ValueError("momentum indices must lie in [0, linear_size)")
    canonical = (2.0 * math.pi / float(linear_size)) * index_array
    centered_indices = index_array.copy()
    centered_indices[centered_indices > linear_size // 2] -= linear_size
    centered = (2.0 * math.pi / float(linear_size)) * centered_indices
    return MomentumGeometry(
        indices=tuple(int(value) for value in index_array),
        canonical_wavevector=canonical,
        centered_wavevector=centered,
        centered_norm=float(np.linalg.norm(centered)),
    )


def dynamic_structure_factor(
    omega: np.ndarray,
    rescaled_spectrum: np.ndarray,
    density_spectrum: np.ndarray,
    *,
    beta: float,
    volume: int,
    particle_count: int,
) -> np.ndarray:
    """Convert per-site MaxEnt output to per-particle positive-frequency S(q,w)."""

    omega = np.asarray(omega, dtype=float)
    rescaled_spectrum = np.asarray(rescaled_spectrum, dtype=float)
    density_spectrum = np.asarray(density_spectrum, dtype=float)
    if (
        omega.ndim != 1
        or omega.size < 2
        or rescaled_spectrum.shape != omega.shape
        or density_spectrum.shape != omega.shape
    ):
        raise ValueError("MaxEnt spectra must share one nontrivial omega grid")
    if (
        np.any(~np.isfinite(omega))
        or np.any(~np.isfinite(rescaled_spectrum))
        or np.any(~np.isfinite(density_spectrum))
        or np.any(np.diff(omega) <= 0.0)
        or omega[0] != 0.0
        or np.any(omega < 0.0)
    ):
        raise ValueError("MaxEnt spectra contain invalid real-frequency data")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    if volume <= 0 or particle_count <= 0:
        raise ValueError("dynamic structure normalization requires positive V and N")
    if np.any(rescaled_spectrum < 0.0) or np.any(density_spectrum < 0.0):
        raise ValueError("MaxEnt density spectra must be nonnegative")

    normalization = float(volume) / float(particle_count)
    result = np.empty_like(omega)
    result[0] = normalization * rescaled_spectrum[0] / beta
    result[1:] = (
        normalization
        * density_spectrum[1:]
        / (-np.expm1(-beta * omega[1:]))
    )
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("dynamic structure factor conversion is non-finite")
    return result


def integrated_static_structure_factor(
    omega: np.ndarray,
    rescaled_spectrum: np.ndarray,
    density_spectrum: np.ndarray,
    *,
    beta: float,
    volume: int,
    particle_count: int,
) -> float:
    """Integrate positive and detailed-balance-reflected continuum spectral weight."""

    structure = dynamic_structure_factor(
        omega,
        rescaled_spectrum,
        density_spectrum,
        beta=beta,
        volume=volume,
        particle_count=particle_count,
    )
    omega = np.asarray(omega, dtype=float)
    density_spectrum = np.asarray(density_spectrum, dtype=float)
    rescaled_spectrum = np.asarray(rescaled_spectrum, dtype=float)
    normalization = float(volume) / float(particle_count)
    integrand = np.empty_like(structure)
    integrand[0] = normalization * 2.0 * rescaled_spectrum[0] / beta
    integrand[1:] = (
        normalization
        * density_spectrum[1:]
        / np.tanh(0.5 * beta * omega[1:])
    )
    value = float(
        np.sum(
            0.5
            * (integrand[:-1] + integrand[1:])
            * np.diff(omega)
        )
    )
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("integrated static structure factor is invalid")
    return value


def jackknife_standard_error(replicates: np.ndarray) -> np.ndarray:
    """Return the delete-one-block jackknife standard error on trailing axes."""

    replicates = np.asarray(replicates, dtype=float)
    if replicates.ndim == 0 or replicates.shape[0] < 2:
        raise ValueError("jackknife requires at least two finite replicates")
    if np.any(~np.isfinite(replicates)):
        raise ValueError("jackknife replicates must be finite")
    count = replicates.shape[0]
    centered = replicates - np.mean(replicates, axis=0)
    variance = ((count - 1.0) / count) * np.sum(centered * centered, axis=0)
    return np.sqrt(np.maximum(0.0, variance))


def _peak_frequency(omega: np.ndarray, structure_factor: np.ndarray) -> float:
    return float(omega[int(np.argmax(structure_factor))])


def _jackknife_reconstruct(
    data: DensityContinuationData,
    blocks: np.ndarray,
    prepared: PreparedMaxEntInput,
    options: MaxEntRunOptions,
    *,
    beta: float,
    volume: int,
    particle_count: int,
) -> JackknifeSummary:
    block_count = blocks.shape[0]
    replicate_means = (
        (float(block_count) * data.means[np.newaxis, :]) - blocks
    ) / float(block_count - 1)
    _, eigenvectors = np.linalg.eigh(data.covariance)
    fixed_transform = (
        eigenvectors[:, prepared.retained_covariance_indices].T
    )
    spectra: list[np.ndarray] = []
    static_values: list[float] = []
    peak_frequencies: list[float] = []
    selected_alphas: list[float] = []
    covariance_chi2: list[float] = []
    converged: list[bool] = []
    for mean in replicate_means:
        replicate_data = replace(data, means=mean)
        replicate_prepared = replace(
            prepared, transformed_data=fixed_transform @ mean
        )
        replicate_run = run_maxent(replicate_data, replicate_prepared, options)
        structure_factor = dynamic_structure_factor(
            replicate_run.omega,
            replicate_run.rescaled_spectrum,
            replicate_run.density_spectrum,
            beta=beta,
            volume=volume,
            particle_count=particle_count,
        )
        spectra.append(structure_factor)
        static_values.append(
            integrated_static_structure_factor(
                replicate_run.omega,
                replicate_run.rescaled_spectrum,
                replicate_run.density_spectrum,
                beta=beta,
                volume=volume,
                particle_count=particle_count,
            )
        )
        peak_frequencies.append(
            _peak_frequency(replicate_run.omega, structure_factor)
        )
        selected_alphas.append(replicate_run.selected_alpha)
        covariance_chi2.append(replicate_run.covariance_chi2)
        converged.append(replicate_run.converged)

    spectra_array = np.asarray(spectra)
    static_array = np.asarray(static_values)
    peak_array = np.asarray(peak_frequencies)
    return JackknifeSummary(
        spectra=spectra_array,
        spectral_standard_error=jackknife_standard_error(spectra_array),
        static_values=static_array,
        static_standard_error=float(jackknife_standard_error(static_array)),
        peak_frequencies=peak_array,
        peak_standard_error=float(jackknife_standard_error(peak_array)),
        selected_alphas=np.asarray(selected_alphas),
        covariance_chi2=np.asarray(covariance_chi2),
        converged=np.asarray(converged, dtype=bool),
    )


def _write_tsv(
    path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _momentum_columns(prefix: str, dimension: int) -> list[str]:
    return [f"{prefix}_{axis}" for axis in range(dimension)]


def _geometry_values(geometry: MomentumGeometry) -> list[float | int]:
    return [
        *geometry.indices,
        *geometry.canonical_wavevector,
        *geometry.centered_wavevector,
        geometry.centered_norm,
    ]


def _write_momentum_outputs(
    directory: Path, result: DynamicStructureResult
) -> None:
    error = (
        result.jackknife.spectral_standard_error
        if result.jackknife is not None
        else np.full_like(result.structure_factor, np.nan)
    )
    _write_tsv(
        directory / "dynamic_structure_factor.tsv",
        (
            "omega",
            "S_q_omega_per_particle",
            "jackknife_standard_error",
            "density_spectral_function_A_per_site",
            "rescaled_spectrum_B_per_site",
        ),
        zip(
            result.maxent.omega,
            result.structure_factor,
            error,
            result.maxent.density_spectrum,
            result.maxent.rescaled_spectrum,
            strict=True,
        ),
    )
    if result.jackknife is None:
        return
    _write_tsv(
        directory / "jackknife_spectra.tsv",
        ("omitted_block", "omega_index", "omega", "S_q_omega_per_particle"),
        (
            (block, point, result.maxent.omega[point], value)
            for block, spectrum in enumerate(result.jackknife.spectra)
            for point, value in enumerate(spectrum)
        ),
    )
    _write_tsv(
        directory / "jackknife_summary.tsv",
        (
            "omitted_block",
            "integrated_static_S_q",
            "peak_omega",
            "selected_alpha",
            "covariance_chi2",
            "converged",
        ),
        (
            (
                block,
                result.jackknife.static_values[block],
                result.jackknife.peak_frequencies[block],
                result.jackknife.selected_alphas[block],
                result.jackknife.covariance_chi2[block],
                int(result.jackknife.converged[block]),
            )
            for block in range(result.jackknife.spectra.shape[0])
        ),
    )


def _write_combined_outputs(
    directory: Path,
    results: Sequence[DynamicStructureResult],
) -> None:
    dimension = len(results[0].momentum.indices)
    geometry_header = [
        *_momentum_columns("k", dimension),
        *_momentum_columns("q_canonical", dimension),
        *_momentum_columns("q_centered", dimension),
        "q_centered_norm",
    ]
    _write_tsv(
        directory / "dynamic_structure_factor.tsv",
        (
            "momentum",
            *geometry_header,
            "omega_index",
            "omega",
            "S_q_omega_per_particle",
            "jackknife_standard_error",
            "density_spectral_function_A_per_site",
            "rescaled_spectrum_B_per_site",
        ),
        (
            (
                result.momentum.ordinal,
                *_geometry_values(result.geometry),
                point,
                result.maxent.omega[point],
                result.structure_factor[point],
                (
                    result.jackknife.spectral_standard_error[point]
                    if result.jackknife is not None
                    else math.nan
                ),
                result.maxent.density_spectrum[point],
                result.maxent.rescaled_spectrum[point],
            )
            for result in results
            for point in range(result.maxent.omega.size)
        ),
    )
    _write_tsv(
        directory / "momentum_summary.tsv",
        (
            "momentum",
            *geometry_header,
            "integrated_static_S_q",
            "static_jackknife_standard_error",
            "peak_omega",
            "peak_jackknife_standard_error",
            "selected_alpha",
            "covariance_chi2",
            "retained_covariance_rank",
            "point_count",
        ),
        (
            (
                result.momentum.ordinal,
                *_geometry_values(result.geometry),
                result.static_structure_factor,
                (
                    result.jackknife.static_standard_error
                    if result.jackknife is not None
                    else math.nan
                ),
                result.peak_frequency,
                (
                    result.jackknife.peak_standard_error
                    if result.jackknife is not None
                    else math.nan
                ),
                result.maxent.selected_alpha,
                result.maxent.covariance_chi2,
                result.prepared.retained_covariance_rank,
                result.data.point_count,
            )
            for result in results
        ),
    )


def _plot_axis(
    results: Sequence[DynamicStructureResult],
) -> tuple[np.ndarray, str, list[str] | None]:
    dimension = len(results[0].momentum.indices)
    if dimension == 1:
        return (
            np.asarray(
                [result.geometry.centered_wavevector[0] for result in results]
            ),
            r"$q$",
            None,
        )
    labels = [
        "(" + ",".join(str(value) for value in result.momentum.indices) + ")"
        for result in results
    ]
    return np.arange(len(results), dtype=float), "momentum index", labels


def _write_plots(
    directory: Path,
    results: Sequence[DynamicStructureResult],
    max_linecuts: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values, x_label, tick_labels = _plot_axis(results)
    order = np.argsort(x_values, kind="stable")
    ordered_results = [results[int(index)] for index in order]
    x_values = x_values[order]
    spectra = np.vstack([result.structure_factor for result in ordered_results])
    omega = ordered_results[0].maxent.omega

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if len(ordered_results) == 1:
        half_width = 0.5
        image = axis.imshow(
            spectra.T,
            origin="lower",
            aspect="auto",
            extent=(
                x_values[0] - half_width,
                x_values[0] + half_width,
                omega[0],
                omega[-1],
            ),
        )
    else:
        image = axis.pcolormesh(
            x_values, omega, spectra.T, shading="nearest"
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(r"$\omega$")
    axis.set_title(r"Dynamic structure factor $S(q,\omega)$")
    if tick_labels is not None:
        axis.set_xticks(x_values, [tick_labels[int(index)] for index in order])
    figure.colorbar(image, ax=axis, label=r"$S(q,\omega)$ per particle")
    figure.savefig(directory / "dynamic_structure_factor_map.png", dpi=180)
    plt.close(figure)

    selected_count = min(max_linecuts, len(ordered_results))
    selected_indices = np.unique(
        np.linspace(0, len(ordered_results) - 1, selected_count, dtype=int)
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for index in selected_indices:
        result = ordered_results[int(index)]
        if len(result.momentum.indices) == 1:
            label = f"q={result.geometry.centered_wavevector[0]:.4g}"
        else:
            label = "k=(" + ",".join(
                str(value) for value in result.momentum.indices
            ) + ")"
        axis.plot(result.maxent.omega, result.structure_factor, label=label)
        if result.jackknife is not None:
            lower = np.maximum(
                0.0,
                result.structure_factor
                - result.jackknife.spectral_standard_error,
            )
            upper = (
                result.structure_factor
                + result.jackknife.spectral_standard_error
            )
            axis.fill_between(
                result.maxent.omega, lower, upper, alpha=0.18
            )
    axis.set_xlabel(r"$\omega$")
    axis.set_ylabel(r"$S(q,\omega)$ per particle")
    axis.set_title("Selected momentum line cuts")
    axis.legend()
    figure.savefig(directory / "dynamic_structure_factor_linecuts.png", dpi=180)
    plt.close(figure)

    peak = np.asarray([result.peak_frequency for result in ordered_results])
    peak_error = np.asarray(
        [
            (
                result.jackknife.peak_standard_error
                if result.jackknife is not None
                else 0.0
            )
            for result in ordered_results
        ]
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.errorbar(x_values, peak, yerr=peak_error, fmt="o-", capsize=3)
    axis.set_xlabel(x_label)
    axis.set_ylabel(r"peak $\omega$")
    axis.set_title("MaxEnt peak dispersion")
    if tick_labels is not None:
        axis.set_xticks(x_values, [tick_labels[int(index)] for index in order])
    figure.savefig(directory / "peak_dispersion.png", dpi=180)
    plt.close(figure)


def _batch_metadata(
    bundle: Path,
    results: Sequence[DynamicStructureResult],
    options: MaxEntRunOptions,
    jackknife: bool,
    max_linecuts: int,
) -> dict[str, Any]:
    first = results[0]
    return {
        "schema": "qmc-dynamic-structure-factor-v1",
        "source_bundle": str(bundle),
        "source_basis": first.data.basis,
        "normalization": "per_particle",
        "dynamic_structure_factor": (
            "S(q,omega)=(V/N)*A_density(q,omega)/(1-exp(-beta*omega))"
        ),
        "omega_zero_limit": "S(q,0)=(V/N)*B(q,0)/beta",
        "static_continuum_sum_rule": (
            "S(q)=(V/N)*integral_0^infinity "
            "A_density(q,omega)*coth(beta*omega/2) d_omega"
        ),
        "elastic_zero_frequency_peak_included": False,
        "wavevector_canonical": "q_alpha=2*pi*k_alpha/L",
        "wavevector_plot_convention": "centered_first_brillouin_zone",
        "momentum_ordinals": [result.momentum.ordinal for result in results],
        "momentum_indices": [
            list(result.momentum.indices) for result in results
        ],
        "jackknife_enabled": jackknife,
        "jackknife_block_count": (
            int(first.jackknife.spectra.shape[0])
            if first.jackknife is not None
            else 0
        ),
        "jackknife_fixed_inputs": [
            "covariance",
            "retained_covariance_eigenmodes",
            "omega_mesh",
            "default_model",
            "alpha_mesh",
            "analyzer",
        ],
        "jackknife_varying_input": "leave_one_block_out_imaginary_axis_mean",
        "omega_max": options.omega_max,
        "omega_points": options.omega_points,
        "alpha_min": options.alpha_min,
        "alpha_max": options.alpha_max,
        "alpha_points": options.alpha_points,
        "covariance_rcond": options.covariance_rcond,
        "max_iterations": options.max_iterations,
        "analyzer": options.analyzer,
        "max_linecuts": max_linecuts,
    }


def run_dynamic_structure_batch(
    bundle: str | Path,
    destination: str | Path,
    options: MaxEntRunOptions,
    *,
    momentum_ordinals: Sequence[int] | None = None,
    jackknife: bool = True,
    max_linecuts: int = 8,
) -> Path:
    """Continue selected momenta and atomically publish combined S(q,w) outputs."""

    options.validate()
    if max_linecuts <= 0:
        raise ValueError("max_linecuts must be positive")
    bundle_path = Path(bundle).resolve()
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"output destination already exists: {destination_path}")

    catalog = list_density_continuation_momenta(bundle_path)
    by_ordinal = {momentum.ordinal: momentum for momentum in catalog}
    if momentum_ordinals is None:
        selected = [
            momentum for momentum in catalog if not momentum.exact_constraint
        ]
    else:
        selected = []
        seen: set[int] = set()
        for ordinal in momentum_ordinals:
            if ordinal in seen:
                raise ValueError(f"duplicate momentum ordinal: {ordinal}")
            seen.add(ordinal)
            try:
                momentum = by_ordinal[ordinal]
            except KeyError as error:
                raise ValueError(f"unknown momentum ordinal: {ordinal}") from error
            if momentum.exact_constraint:
                raise ValueError(
                    f"momentum ordinal {ordinal} is an exact q=0 constraint"
                )
            selected.append(momentum)
    if not selected:
        raise ValueError("no measured nonzero momenta were selected")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f"{destination_path.name}.tmp.", dir=destination_path.parent
        )
    )
    try:
        results: list[DynamicStructureResult] = []
        for momentum in selected:
            if not options.quiet:
                print(
                    f"continuing momentum {momentum.ordinal} "
                    f"k={momentum.indices}"
                )
            data = load_density_continuation_bundle(
                bundle_path, momentum.ordinal
            )
            particle_count = _manifest_nonnegative_int(
                data.manifest, "model_particle_count"
            )
            volume = _manifest_nonnegative_int(
                data.manifest, "lattice_volume"
            )
            linear_size = _manifest_nonnegative_int(
                data.manifest, "model_linear_size"
            )
            if particle_count == 0 or volume == 0 or linear_size == 0:
                raise BundleFormatError(
                    "dynamic structure factor requires positive N, V, and L"
                )
            geometry = momentum_geometry(momentum.indices, linear_size)
            prepared = prepare_maxent_input(data, options)
            maxent = run_maxent(data, prepared, options)
            structure_factor = dynamic_structure_factor(
                maxent.omega,
                maxent.rescaled_spectrum,
                maxent.density_spectrum,
                beta=data.beta,
                volume=volume,
                particle_count=particle_count,
            )
            static_structure_factor = integrated_static_structure_factor(
                maxent.omega,
                maxent.rescaled_spectrum,
                maxent.density_spectrum,
                beta=data.beta,
                volume=volume,
                particle_count=particle_count,
            )
            jackknife_result = None
            if jackknife:
                jackknife_result = _jackknife_reconstruct(
                    data,
                    load_density_block_values(data),
                    prepared,
                    options,
                    beta=data.beta,
                    volume=volume,
                    particle_count=particle_count,
                )
            result = DynamicStructureResult(
                momentum=momentum,
                geometry=geometry,
                data=data,
                prepared=prepared,
                maxent=maxent,
                structure_factor=structure_factor,
                static_structure_factor=static_structure_factor,
                peak_frequency=_peak_frequency(maxent.omega, structure_factor),
                jackknife=jackknife_result,
            )
            momentum_directory = (
                temporary / f"momentum-{momentum.ordinal:04d}"
            )
            write_adapter_output(
                momentum_directory, data, prepared, options, maxent
            )
            _write_momentum_outputs(momentum_directory, result)
            results.append(result)

        _write_combined_outputs(temporary, results)
        _write_plots(temporary, results, max_linecuts)
        with (temporary / "batch_run.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                _batch_metadata(
                    bundle_path, results, options, jackknife, max_linecuts
                ),
                stream,
                indent=2,
            )
            stream.write("\n")
        os.replace(temporary, destination_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination_path


def _parse_momentum_ordinals(value: str) -> list[int]:
    if not value.strip():
        raise argparse.ArgumentTypeError("momentum ordinals must not be empty")
    ordinals: list[int] = []
    for item in value.split(","):
        try:
            ordinal = int(item.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "momentum ordinals must be comma-separated integers"
            ) from error
        if ordinal < 0:
            raise argparse.ArgumentTypeError(
                "momentum ordinals must be nonnegative"
            )
        ordinals.append(ordinal)
    return ordinals


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch continue every selected nonzero density momentum, convert "
            "to per-particle S(q,omega), propagate block jackknife errors, "
            "and write combined tables and plots."
        )
    )
    parser.add_argument("bundle", help="density-continuation-v1 directory")
    parser.add_argument(
        "--output-dir", required=True, help="new batch result directory"
    )
    parser.add_argument(
        "--momentum-ordinals",
        type=_parse_momentum_ordinals,
        help=(
            "optional comma-separated momentum ordinals; the default is every "
            "measured nonzero momentum"
        ),
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        required=True,
        help="positive real-frequency cutoff; this is an analysis choice",
    )
    parser.add_argument("--omega-points", type=int, default=200)
    parser.add_argument("--alpha-min", type=float, default=1.0e-4)
    parser.add_argument("--alpha-max", type=float, default=20.0)
    parser.add_argument("--alpha-points", type=int, default=20)
    parser.add_argument("--covariance-rcond", type=float, default=1.0e-12)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument(
        "--analyzer",
        choices=(
            "LineFitAnalyzer",
            "Chi2CurvatureAnalyzer",
            "EntropyAnalyzer",
        ),
        default="LineFitAnalyzer",
    )
    parser.add_argument(
        "--no-jackknife",
        action="store_true",
        help="skip leave-one-block-out MaxEnt reconstructions",
    )
    parser.add_argument(
        "--max-linecuts",
        type=int,
        default=8,
        help="maximum number of momentum curves in the line-cut plot",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    options = MaxEntRunOptions(
        omega_max=arguments.omega_max,
        omega_points=arguments.omega_points,
        alpha_min=arguments.alpha_min,
        alpha_max=arguments.alpha_max,
        alpha_points=arguments.alpha_points,
        covariance_rcond=arguments.covariance_rcond,
        max_iterations=arguments.max_iterations,
        analyzer=arguments.analyzer,
        quiet=arguments.quiet,
    )
    try:
        output = run_dynamic_structure_batch(
            arguments.bundle,
            arguments.output_dir,
            options,
            momentum_ordinals=arguments.momentum_ordinals,
            jackknife=not arguments.no_jackknife,
            max_linecuts=arguments.max_linecuts,
        )
    except (BundleFormatError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(f"Dynamic structure-factor result written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
