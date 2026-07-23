"""TRIQS/maxent adapter for density-continuation-v1 bundles.

The QMC bundle stores a positive density spectrum convention

    A(w) = -Im chi^R(w) / pi,  w > 0.

Both supported kernels are singular as 1/w at the origin.  This adapter asks
MaxEnt to continue the regularized positive spectrum B(w) = A(w) / w instead:

    chi(iwn) = integral 2 w^2 / (w^2 + wn^2) B(w) dw
    C(tau)   = integral w cosh((beta/2-tau)w) / sinh(beta*w/2) B(w) dw.

The output contains both B and the physical A = w B.  An elastic delta
function at w=0 is not represented by this regular continuum mesh.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import tempfile
import types
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import nnls


TRIQS_MAXENT_VERSION = "4.0.0"
TRIQS_MAXENT_COMMIT = "79926af3a6e49310570839d19cf3bc1a917778c2"
SCHEMA_ID = "density-continuation"
SCHEMA_VERSION = "1"
KERNEL_CONVENTION = "positive_density_spectrum_bosonic_v1"
_VENDORED_PACKAGE_NAME = "_qmc_vendored_triqs_maxent"


class BundleFormatError(ValueError):
    """Raised when a continuation bundle is incomplete or inconsistent."""


@dataclass(frozen=True)
class DensityContinuationData:
    bundle: Path
    manifest: Mapping[str, str]
    basis: str
    beta: float
    momentum_ordinal: int
    momentum_indices: tuple[int, ...]
    point_ordinals: np.ndarray
    point_indices: np.ndarray
    coordinates: np.ndarray
    means: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.means.size)


@dataclass(frozen=True)
class PreparedMaxEntInput:
    omega: np.ndarray
    kernel: np.ndarray
    transformed_kernel: np.ndarray
    transformed_data: np.ndarray
    transformed_errors: np.ndarray
    covariance_eigenvalues: np.ndarray
    retained_covariance_indices: np.ndarray
    default_model_normalization: float
    default_model_normalization_source: str

    @property
    def retained_covariance_rank(self) -> int:
        return int(self.retained_covariance_indices.size)


@dataclass(frozen=True)
class MaxEntRunOptions:
    omega_max: float
    omega_points: int = 200
    alpha_min: float = 1.0e-4
    alpha_max: float = 20.0
    alpha_points: int = 20
    covariance_rcond: float = 1.0e-12
    max_iterations: int = 1000
    analyzer: str = "LineFitAnalyzer"
    quiet: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.omega_max) or self.omega_max <= 0.0:
            raise ValueError("omega_max must be finite and positive")
        if self.omega_points < 3:
            raise ValueError("omega_points must be at least 3")
        if (
            not math.isfinite(self.alpha_min)
            or not math.isfinite(self.alpha_max)
            or self.alpha_min <= 0.0
            or self.alpha_max <= self.alpha_min
        ):
            raise ValueError("alpha bounds must satisfy 0 < alpha_min < alpha_max")
        if self.alpha_points < 5:
            raise ValueError("alpha_points must be at least 5")
        if (
            not math.isfinite(self.covariance_rcond)
            or self.covariance_rcond < 0.0
            or self.covariance_rcond >= 1.0
        ):
            raise ValueError("covariance_rcond must lie in [0, 1)")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.analyzer not in {
            "LineFitAnalyzer",
            "Chi2CurvatureAnalyzer",
            "EntropyAnalyzer",
        }:
            raise ValueError(f"unsupported analyzer: {self.analyzer}")


@dataclass(frozen=True)
class MaxEntRun:
    omega: np.ndarray
    rescaled_spectrum: np.ndarray
    density_spectrum: np.ndarray
    default_model: np.ndarray
    reconstruction: np.ndarray
    alphas: np.ndarray
    chi2: np.ndarray
    entropy: np.ndarray
    cost: np.ndarray
    analyzer: str
    selected_alpha_index: int
    selected_alpha: float
    selected_chi2: float
    covariance_chi2: float
    converged: bool


def _read_tsv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is None:
                raise BundleFormatError(f"{path} has no TSV header")
            rows = list(reader)
    except OSError as error:
        raise BundleFormatError(f"cannot read {path}: {error}") from error
    if not rows:
        raise BundleFormatError(f"{path} contains no data rows")
    return rows


def _read_manifest(path: Path) -> dict[str, str]:
    rows = _read_tsv(path)
    manifest: dict[str, str] = {}
    for row in rows:
        if set(row) != {"key", "value"}:
            raise BundleFormatError(f"{path} must have exactly key and value columns")
        key = row["key"]
        if not key or key in manifest:
            raise BundleFormatError(f"{path} contains an empty or duplicate key: {key!r}")
        manifest[key] = row["value"]
    return manifest


def _required(manifest: Mapping[str, str], key: str) -> str:
    try:
        return manifest[key]
    except KeyError as error:
        raise BundleFormatError(f"manifest is missing required key {key!r}") from error


def _manifest_int(manifest: Mapping[str, str], key: str) -> int:
    value = _required(manifest, key)
    try:
        parsed = int(value)
    except ValueError as error:
        raise BundleFormatError(f"manifest key {key!r} is not an integer") from error
    if parsed < 0:
        raise BundleFormatError(f"manifest key {key!r} must be nonnegative")
    return parsed


def _manifest_float(manifest: Mapping[str, str], key: str) -> float:
    value = _required(manifest, key)
    try:
        parsed = float(value)
    except ValueError as error:
        raise BundleFormatError(f"manifest key {key!r} is not numeric") from error
    if not math.isfinite(parsed):
        raise BundleFormatError(f"manifest key {key!r} must be finite")
    return parsed


def _parse_int(row: Mapping[str, str], key: str, table: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, ValueError) as error:
        raise BundleFormatError(f"{table} has an invalid {key!r} value") from error
    return value


def _parse_float(row: Mapping[str, str], key: str, table: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise BundleFormatError(f"{table} has an invalid {key!r} value") from error
    if not math.isfinite(value):
        raise BundleFormatError(f"{table} has a non-finite {key!r} value")
    return value


def _validate_manifest(manifest: Mapping[str, str]) -> tuple[str, float]:
    expected = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "observable_id": "connected_density_per_site",
        "kernel_convention_id": KERNEL_CONVENTION,
    }
    for key, value in expected.items():
        if _required(manifest, key) != value:
            raise BundleFormatError(
                f"unsupported manifest {key}: {_required(manifest, key)!r}"
            )
    basis = _required(manifest, "basis")
    if basis not in {"bosonic_matsubara", "imaginary_time_lag"}:
        raise BundleFormatError(f"unsupported continuation basis: {basis!r}")
    beta = _manifest_float(manifest, "model_beta")
    if beta <= 0.0:
        raise BundleFormatError("manifest model_beta must be positive")
    return basis, beta


def load_density_continuation_bundle(
    bundle: str | Path, momentum_ordinal: int = 0
) -> DensityContinuationData:
    """Load and validate one momentum from a density-continuation-v1 bundle."""

    bundle_path = Path(bundle).resolve()
    if momentum_ordinal < 0:
        raise BundleFormatError("momentum ordinal must be nonnegative")
    manifest = _read_manifest(bundle_path / "manifest.tsv")
    basis, beta = _validate_manifest(manifest)
    momentum_count = _manifest_int(manifest, "momentum_count")
    if momentum_ordinal >= momentum_count:
        raise BundleFormatError(
            f"momentum ordinal {momentum_ordinal} is outside [0, {momentum_count})"
        )

    values_rows = _read_tsv(bundle_path / "values.tsv")
    selected = [
        row
        for row in values_rows
        if _parse_int(row, "momentum", "values.tsv") == momentum_ordinal
    ]
    point_column = "frequency" if basis == "bosonic_matsubara" else "lag"
    index_column = "n" if basis == "bosonic_matsubara" else "lag"
    coordinate_column = "omega" if basis == "bosonic_matsubara" else "tau"
    selected.sort(key=lambda row: _parse_int(row, point_column, "values.tsv"))
    expected_point_count = _manifest_int(
        manifest, "frequency_count" if basis == "bosonic_matsubara" else "lag_count"
    )
    if len(selected) != expected_point_count:
        raise BundleFormatError(
            f"values.tsv has {len(selected)} selected rows, expected {expected_point_count}"
        )

    point_ordinals = np.array(
        [_parse_int(row, point_column, "values.tsv") for row in selected], dtype=int
    )
    if not np.array_equal(point_ordinals, np.arange(expected_point_count)):
        raise BundleFormatError("selected values.tsv point ordinals are not contiguous")
    if any(_parse_int(row, "exact_constraint", "values.tsv") != 0 for row in selected):
        raise BundleFormatError(
            "MaxEnt requires measured nonzero-momentum data, not exact constraints"
        )

    if basis == "bosonic_matsubara":
        point_indices = np.array(
            [_parse_int(row, index_column, "values.tsv") for row in selected],
            dtype=int,
        )
    else:
        point_indices = point_ordinals.copy()
    coordinates = np.array(
        [_parse_float(row, coordinate_column, "values.tsv") for row in selected]
    )
    means = np.array([_parse_float(row, "mean", "values.tsv") for row in selected])
    standard_errors = np.array(
        [_parse_float(row, "standard_error", "values.tsv") for row in selected]
    )
    if np.any(standard_errors < 0.0):
        raise BundleFormatError("values.tsv contains a negative standard error")
    if basis == "bosonic_matsubara":
        if np.any(point_indices < 0):
            raise BundleFormatError("MaxEnt adapter requires nonnegative Matsubara indices")
        expected_omega = (2.0 * math.pi / beta) * point_indices
        if not np.allclose(coordinates, expected_omega, rtol=1.0e-12, atol=1.0e-14):
            raise BundleFormatError("values.tsv Matsubara frequencies disagree with beta and n")
    elif np.any(coordinates < 0.0) or np.any(coordinates >= beta):
        raise BundleFormatError("values.tsv lags must lie in [0, beta)")

    momentum_columns = sorted(
        (key for key in selected[0] if key.startswith("k_")),
        key=lambda key: int(key[2:]),
    )
    dimension = _manifest_int(manifest, "model_dimension")
    if len(momentum_columns) != dimension:
        raise BundleFormatError("values.tsv momentum components disagree with model dimension")
    momentum_indices = tuple(
        _parse_int(selected[0], key, "values.tsv") for key in momentum_columns
    )
    for row in selected[1:]:
        if tuple(_parse_int(row, key, "values.tsv") for key in momentum_columns) != momentum_indices:
            raise BundleFormatError("values.tsv selected momentum components are inconsistent")
    if all(component == 0 for component in momentum_indices):
        raise BundleFormatError("fixed-particle-number q=0 density data cannot be continued")

    covariance = np.empty((expected_point_count, expected_point_count), dtype=float)
    seen = np.zeros_like(covariance, dtype=bool)
    covariance_rows = _read_tsv(bundle_path / "covariance.tsv")
    for row in covariance_rows:
        if _parse_int(row, "momentum", "covariance.tsv") != momentum_ordinal:
            continue
        left = _parse_int(row, "left", "covariance.tsv")
        right = _parse_int(row, "right", "covariance.tsv")
        if not (0 <= left < expected_point_count and 0 <= right < expected_point_count):
            raise BundleFormatError("covariance.tsv contains an out-of-range matrix index")
        if seen[left, right]:
            raise BundleFormatError("covariance.tsv contains a duplicate matrix entry")
        covariance[left, right] = _parse_float(
            row, "covariance_of_mean", "covariance.tsv"
        )
        seen[left, right] = True
    if not np.all(seen):
        raise BundleFormatError("covariance.tsv does not contain a complete selected matrix")
    covariance_scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(
        covariance, covariance.T, rtol=1.0e-12, atol=1.0e-14 * covariance_scale
    ):
        raise BundleFormatError("selected covariance matrix is not symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    if not np.allclose(
        np.diag(covariance),
        standard_errors**2,
        rtol=2.0e-10,
        atol=1.0e-14 * covariance_scale,
    ):
        raise BundleFormatError(
            "values.tsv standard errors disagree with the covariance diagonal"
        )
    eigenvalues = np.linalg.eigvalsh(covariance)
    psd_tolerance = max(1.0e-15, covariance_scale * 1.0e-11)
    if eigenvalues[0] < -psd_tolerance:
        raise BundleFormatError("selected covariance matrix is not positive semidefinite")

    return DensityContinuationData(
        bundle=bundle_path,
        manifest=manifest,
        basis=basis,
        beta=beta,
        momentum_ordinal=momentum_ordinal,
        momentum_indices=momentum_indices,
        point_ordinals=point_ordinals,
        point_indices=point_indices,
        coordinates=coordinates,
        means=means,
        standard_errors=standard_errors,
        covariance=covariance,
    )


def regularized_bosonic_kernel(
    basis: str, beta: float, coordinates: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    """Return the kernel for B(omega)=A(omega)/omega, including omega=0."""

    coordinates = np.asarray(coordinates, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    if omega.ndim != 1 or omega.size < 2 or np.any(~np.isfinite(omega)):
        raise ValueError("omega must be a finite one-dimensional grid")
    if np.any(omega < 0.0) or np.any(np.diff(omega) <= 0.0):
        raise ValueError("omega must be strictly increasing and nonnegative")
    if coordinates.ndim != 1 or np.any(~np.isfinite(coordinates)):
        raise ValueError("coordinates must be a finite one-dimensional array")

    omega_rows = omega[np.newaxis, :]
    if basis == "bosonic_matsubara":
        frequencies = coordinates[:, np.newaxis]
        denominator = omega_rows**2 + frequencies**2
        kernel = np.divide(
            2.0 * omega_rows**2,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0.0,
        )
        zero_frequency = frequencies[:, 0] == 0.0
        zero_omega = omega == 0.0
        kernel[np.ix_(zero_frequency, zero_omega)] = 2.0
        return kernel
    if basis == "imaginary_time_lag":
        tau = coordinates[:, np.newaxis]
        if np.any(tau < 0.0) or np.any(tau >= beta):
            raise ValueError("imaginary-time lags must lie in [0, beta)")
        denominator = -np.expm1(-beta * omega_rows)
        numerator = np.exp(-tau * omega_rows) + np.exp(-(beta - tau) * omega_rows)
        kernel = np.divide(
            omega_rows * numerator,
            denominator,
            out=np.empty_like(numerator),
            where=denominator != 0.0,
        )
        kernel[:, omega == 0.0] = 2.0 / beta
        return kernel
    raise ValueError(f"unsupported continuation basis: {basis!r}")


def _linear_grid_delta(omega: np.ndarray) -> np.ndarray:
    delta = np.empty_like(omega)
    delta[1:-1] = 0.5 * (omega[2:] - omega[:-2])
    delta[0] = 0.5 * (omega[1] - omega[0])
    delta[-1] = 0.5 * (omega[-1] - omega[-2])
    return delta


def prepare_maxent_input(
    data: DensityContinuationData, options: MaxEntRunOptions
) -> PreparedMaxEntInput:
    """Build the bosonic DataKernel and diagonalize sampling covariance."""

    options.validate()
    omega = np.linspace(0.0, options.omega_max, options.omega_points)
    kernel = regularized_bosonic_kernel(
        data.basis, data.beta, data.coordinates, omega
    )

    eigenvalues, eigenvectors = np.linalg.eigh(data.covariance)
    largest = float(eigenvalues[-1])
    if not math.isfinite(largest) or largest <= 0.0:
        raise ValueError("covariance has no positive statistical mode")
    negative_tolerance = max(1.0e-15, largest * 1.0e-10)
    if eigenvalues[0] < -negative_tolerance:
        raise ValueError("covariance has a materially negative eigenvalue")
    threshold = largest * options.covariance_rcond
    retained = np.flatnonzero(eigenvalues > max(0.0, threshold))
    if retained.size == 0:
        raise ValueError("covariance threshold removed every statistical mode")
    transform = eigenvectors[:, retained].T
    transformed_data = transform @ data.means
    transformed_kernel = transform @ kernel
    transformed_errors = np.sqrt(eigenvalues[retained])

    normalization_source = "weighted_nonnegative_least_squares"
    zero_frequency = (
        np.flatnonzero(data.point_indices == 0)
        if data.basis == "bosonic_matsubara"
        else np.empty(0, dtype=int)
    )
    if zero_frequency.size == 1 and data.means[zero_frequency[0]] > 0.0:
        normalization = float(data.means[zero_frequency[0]] / 2.0)
        normalization_source = "matsubara_zero_mode_sum_rule"
    else:
        weighted_kernel = transformed_kernel / transformed_errors[:, np.newaxis]
        weighted_data = transformed_data / transformed_errors
        spectral_weights, _ = nnls(weighted_kernel, weighted_data)
        normalization = float(np.sum(spectral_weights))
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("could not infer a positive default-model normalization")

    return PreparedMaxEntInput(
        omega=omega,
        kernel=kernel,
        transformed_kernel=transformed_kernel,
        transformed_data=transformed_data,
        transformed_errors=transformed_errors,
        covariance_eigenvalues=eigenvalues,
        retained_covariance_indices=retained,
        default_model_normalization=normalization,
        default_model_normalization_source=normalization_source,
    )


def _install_triqs_import_stubs() -> dict[str, types.ModuleType | None]:
    """Install minimal temporary modules needed by an unused upstream helper."""

    names = ("triqs", "triqs.gfs", "triqs.utility", "triqs.utility.mpi")
    previous = {name: sys.modules.get(name) for name in names}
    triqs = types.ModuleType("triqs")
    triqs.__path__ = []  # type: ignore[attr-defined]
    gfs = types.ModuleType("triqs.gfs")
    utility = types.ModuleType("triqs.utility")
    utility.__path__ = []  # type: ignore[attr-defined]
    mpi = types.ModuleType("triqs.utility.mpi")
    utility.mpi = mpi  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "triqs": triqs,
            "triqs.gfs": gfs,
            "triqs.utility": utility,
            "triqs.utility.mpi": mpi,
        }
    )
    return previous


def _restore_modules(previous: Mapping[str, types.ModuleType | None]) -> None:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def load_vendored_maxent_core() -> Any:
    """Load the vendored array-based MaxEnt core without importing TRIQS GFs."""

    cached = sys.modules.get(_VENDORED_PACKAGE_NAME)
    if cached is None:
        package_directory = (
            Path(__file__).resolve().parents[1]
            / "vendor"
            / "triqs_maxent"
            / "python"
            / "triqs_maxent"
        )
        if not (package_directory / "maxent_loop.py").is_file():
            raise RuntimeError(f"vendored TRIQS/maxent source not found at {package_directory}")
        package = types.ModuleType(_VENDORED_PACKAGE_NAME)
        package.__path__ = [str(package_directory)]  # type: ignore[attr-defined]
        package.__package__ = _VENDORED_PACKAGE_NAME
        sys.modules[_VENDORED_PACKAGE_NAME] = package

    previous: dict[str, types.ModuleType | None] | None = None
    try:
        importlib.import_module("triqs.gfs")
    except ModuleNotFoundError as error:
        if error.name not in {"triqs", "triqs.gfs"}:
            raise
        previous = _install_triqs_import_stubs()

    try:
        return types.SimpleNamespace(
            kernels=importlib.import_module(f"{_VENDORED_PACKAGE_NAME}.kernels"),
            omega_meshes=importlib.import_module(
                f"{_VENDORED_PACKAGE_NAME}.omega_meshes"
            ),
            default_models=importlib.import_module(
                f"{_VENDORED_PACKAGE_NAME}.default_models"
            ),
            cost_functions=importlib.import_module(
                f"{_VENDORED_PACKAGE_NAME}.cost_functions"
            ),
            alpha_meshes=importlib.import_module(
                f"{_VENDORED_PACKAGE_NAME}.alpha_meshes"
            ),
            minimizers=importlib.import_module(
                f"{_VENDORED_PACKAGE_NAME}.minimizers"
            ),
            maxent_loop=importlib.import_module(
                f"{_VENDORED_PACKAGE_NAME}.maxent_loop"
            ),
            logtaker=importlib.import_module(f"{_VENDORED_PACKAGE_NAME}.logtaker"),
            analyzers=importlib.import_module(
                f"{_VENDORED_PACKAGE_NAME}.analyzers"
            ),
        )
    finally:
        if previous is not None:
            _restore_modules(previous)


def run_maxent(
    data: DensityContinuationData,
    prepared: PreparedMaxEntInput,
    options: MaxEntRunOptions,
) -> MaxEntRun:
    """Run the vendored TRIQS/maxent optimizer on prepared density data."""

    options.validate()
    core = load_vendored_maxent_core()
    omega = core.omega_meshes.LinearOmegaMesh(
        omega_min=0.0,
        omega_max=options.omega_max,
        n_points=options.omega_points,
    )
    if not np.array_equal(np.asarray(omega), prepared.omega):
        raise RuntimeError("vendored omega mesh disagrees with the prepared grid")

    kernel = core.kernels.DataKernel(
        np.arange(prepared.retained_covariance_rank, dtype=float),
        omega,
        prepared.transformed_kernel,
    )
    default_density = np.full(
        options.omega_points,
        prepared.default_model_normalization / float(np.sum(omega.delta)),
    )
    default_model = core.default_models.DataDefaultModel(
        default_density, omega, omega
    )
    cost_function = core.cost_functions.BryanCostFunction()
    cost_function.K = kernel
    cost_function.D = default_model
    cost_function.G = prepared.transformed_data
    cost_function.err = prepared.transformed_errors

    analyzer_type = getattr(core.analyzers, options.analyzer)
    alpha_mesh = core.alpha_meshes.LogAlphaMesh(
        alpha_min=options.alpha_min,
        alpha_max=options.alpha_max,
        n_points=options.alpha_points,
    )
    minimizer = core.minimizers.LevenbergMinimizer(
        maxiter=options.max_iterations
    )
    loop = core.maxent_loop.MaxEntLoop(
        cost_function=cost_function,
        minimizer=minimizer,
        alpha_mesh=alpha_mesh,
        analyzers=[analyzer_type()],
        interactive=False,
        scale_alpha=None,
    )
    verbosity = (
        core.logtaker.VerbosityFlags.Quiet
        if options.quiet
        else (
            core.logtaker.VerbosityFlags.Header
            | core.logtaker.VerbosityFlags.AlphaLoop
            | core.logtaker.VerbosityFlags.Timing
            | core.logtaker.VerbosityFlags.Errors
        )
    )
    loop.set_verbosity(verbosity=verbosity)
    result = loop.run()
    if result is None:
        raise RuntimeError("TRIQS/maxent declined to run because all input values are tiny")

    analyzer_result = result.analyzer_results[options.analyzer]
    rescaled_spectrum = np.asarray(analyzer_result["A_out"], dtype=float)
    selected_alpha_index = int(analyzer_result["alpha_index"])
    if (
        rescaled_spectrum.shape != prepared.omega.shape
        or np.any(~np.isfinite(rescaled_spectrum))
        or np.any(rescaled_spectrum < 0.0)
    ):
        raise RuntimeError("TRIQS/maxent returned an invalid positive spectrum")
    density_spectrum = prepared.omega * rescaled_spectrum
    reconstruction = prepared.kernel @ (
        rescaled_spectrum * _linear_grid_delta(prepared.omega)
    )
    residual = data.means - reconstruction
    retained = prepared.retained_covariance_indices
    _, eigenvectors = np.linalg.eigh(data.covariance)
    transformed_residual = eigenvectors[:, retained].T @ residual
    covariance_chi2 = float(
        np.sum(
            transformed_residual**2
            / prepared.covariance_eigenvalues[retained]
        )
    )
    alphas = np.asarray(result.alpha, dtype=float)
    chi2 = np.asarray(result.chi2, dtype=float)
    entropy = np.asarray(result.S, dtype=float)
    cost = np.asarray(result.Q, dtype=float)
    converged = bool(
        bool(np.all(np.isfinite(chi2))) and bool(minimizer.converged)
    )

    return MaxEntRun(
        omega=prepared.omega,
        rescaled_spectrum=rescaled_spectrum,
        density_spectrum=density_spectrum,
        default_model=default_density,
        reconstruction=reconstruction,
        alphas=alphas,
        chi2=chi2,
        entropy=entropy,
        cost=cost,
        analyzer=options.analyzer,
        selected_alpha_index=selected_alpha_index,
        selected_alpha=float(alphas[selected_alpha_index]),
        selected_chi2=float(chi2[selected_alpha_index]),
        covariance_chi2=covariance_chi2,
        converged=converged,
    )


def _write_tsv(path: Path, header: Sequence[str], columns: Sequence[np.ndarray]) -> None:
    matrix = np.column_stack(columns)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(header) + "\n")
        np.savetxt(stream, matrix, delimiter="\t", fmt="%.17g")


def _metadata(
    data: DensityContinuationData,
    prepared: PreparedMaxEntInput,
    options: MaxEntRunOptions,
    run: MaxEntRun | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "adapter_schema": "qmc-triqs-maxent-v1",
        "source_bundle": str(data.bundle),
        "source_schema_id": SCHEMA_ID,
        "source_schema_version": SCHEMA_VERSION,
        "source_basis": data.basis,
        "source_kernel_convention": KERNEL_CONVENTION,
        "momentum_ordinal": data.momentum_ordinal,
        "momentum_indices": list(data.momentum_indices),
        "beta": data.beta,
        "spectrum_parameterization": "B(omega)=A_density(omega)/omega",
        "physical_spectrum": "A_density(omega)=omega*B(omega)",
        "elastic_zero_frequency_peak_included": False,
        "triqs_maxent_version": TRIQS_MAXENT_VERSION,
        "triqs_maxent_commit": TRIQS_MAXENT_COMMIT,
        "omega_max": options.omega_max,
        "omega_points": options.omega_points,
        "alpha_min": options.alpha_min,
        "alpha_max": options.alpha_max,
        "alpha_points": options.alpha_points,
        "max_iterations": options.max_iterations,
        "analyzer": options.analyzer,
        "covariance_rcond": options.covariance_rcond,
        "covariance_eigenvalues": prepared.covariance_eigenvalues.tolist(),
        "retained_covariance_indices": prepared.retained_covariance_indices.tolist(),
        "retained_covariance_rank": prepared.retained_covariance_rank,
        "dropped_covariance_modes": (
            data.point_count - prepared.retained_covariance_rank
        ),
        "default_model_normalization": prepared.default_model_normalization,
        "default_model_normalization_source": (
            prepared.default_model_normalization_source
        ),
        "status": "prepared" if run is None else "completed",
    }
    if run is not None:
        metadata.update(
            {
                "selected_alpha_index": run.selected_alpha_index,
                "selected_alpha": run.selected_alpha,
                "selected_chi2_in_covariance_basis": run.selected_chi2,
                "reconstructed_covariance_chi2": run.covariance_chi2,
                "minimizer_last_alpha_converged": run.converged,
            }
        )
    return metadata


def write_adapter_output(
    destination: str | Path,
    data: DensityContinuationData,
    prepared: PreparedMaxEntInput,
    options: MaxEntRunOptions,
    run: MaxEntRun | None,
) -> Path:
    """Atomically write converted arrays and optional MaxEnt results."""

    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"output destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f"{destination_path.name}.tmp.", dir=destination_path.parent
        )
    )
    try:
        coordinate_name = "omega_n" if data.basis == "bosonic_matsubara" else "tau"
        _write_tsv(
            temporary / "input_data.tsv",
            ("point", coordinate_name, "mean", "standard_error"),
            (
                data.point_ordinals,
                data.coordinates,
                data.means,
                data.standard_errors,
            ),
        )
        np.savetxt(
            temporary / "covariance.txt", data.covariance, fmt="%.17g"
        )
        np.savetxt(temporary / "omega.txt", prepared.omega, fmt="%.17g")
        np.savetxt(temporary / "kernel.txt", prepared.kernel, fmt="%.17g")
        np.savetxt(
            temporary / "transformed_kernel.txt",
            prepared.transformed_kernel,
            fmt="%.17g",
        )
        _write_tsv(
            temporary / "transformed_data.tsv",
            ("covariance_mode", "value", "standard_error"),
            (
                prepared.retained_covariance_indices,
                prepared.transformed_data,
                prepared.transformed_errors,
            ),
        )
        if run is not None:
            _write_tsv(
                temporary / "spectrum.tsv",
                (
                    "omega",
                    "rescaled_spectrum_B",
                    "density_spectral_function_A",
                    "default_model_B",
                ),
                (
                    run.omega,
                    run.rescaled_spectrum,
                    run.density_spectrum,
                    run.default_model,
                ),
            )
            _write_tsv(
                temporary / "reconstruction.tsv",
                (
                    "point",
                    coordinate_name,
                    "observed",
                    "standard_error",
                    "reconstructed",
                    "residual",
                ),
                (
                    data.point_ordinals,
                    data.coordinates,
                    data.means,
                    data.standard_errors,
                    run.reconstruction,
                    data.means - run.reconstruction,
                ),
            )
            _write_tsv(
                temporary / "alpha_diagnostics.tsv",
                ("alpha", "chi2", "entropy", "cost"),
                (run.alphas, run.chi2, run.entropy, run.cost),
            )
        with (temporary / "run.json").open("w", encoding="utf-8") as stream:
            json.dump(_metadata(data, prepared, options, run), stream, indent=2)
            stream.write("\n")
        os.replace(temporary, destination_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination_path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a density-continuation-v1 bundle and run the vendored "
            "TRIQS/maxent DataKernel on its bosonic density response."
        )
    )
    parser.add_argument("bundle", help="density-continuation-v1 directory")
    parser.add_argument(
        "--output-dir", required=True, help="new adapter/result directory"
    )
    parser.add_argument(
        "--momentum-ordinal",
        type=int,
        default=0,
        help="zero-based momentum row ordinal (default: 0)",
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
        "--prepare-only",
        action="store_true",
        help="write converted arrays without running MaxEnt",
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
        data = load_density_continuation_bundle(
            arguments.bundle, arguments.momentum_ordinal
        )
        prepared = prepare_maxent_input(data, options)
        run = None if arguments.prepare_only else run_maxent(data, prepared, options)
        output = write_adapter_output(
            arguments.output_dir, data, prepared, options, run
        )
    except (BundleFormatError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(f"MaxEnt {'input' if run is None else 'result'} written to {output}")
    print(
        "covariance modes retained = "
        f"{prepared.retained_covariance_rank}/{data.point_count}"
    )
    if run is not None:
        print(
            f"{run.analyzer}: alpha={run.selected_alpha:.8g}, "
            f"chi2={run.covariance_chi2:.8g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
