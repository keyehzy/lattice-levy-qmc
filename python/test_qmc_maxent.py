from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from qmc_maxent import (
    BundleFormatError,
    MaxEntRunOptions,
    load_density_continuation_bundle,
    prepare_maxent_input,
    regularized_bosonic_kernel,
    run_maxent,
    write_adapter_output,
)


def _write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _synthetic_bundle(path: Path, basis: str = "bosonic_matsubara") -> Path:
    path.mkdir()
    beta = 1.0
    point_count = 6
    manifest = {
        "schema_id": "density-continuation",
        "schema_version": "1",
        "basis": basis,
        "observable_id": "connected_density_per_site",
        "kernel_convention_id": "positive_density_spectrum_bosonic_v1",
        "model_beta": str(beta),
        "model_dimension": "1",
        "momentum_count": "1",
        "frequency_count" if basis == "bosonic_matsubara" else "lag_count": str(
            point_count
        ),
    }
    _write_tsv(
        path / "manifest.tsv",
        ["key", "value"],
        [[key, value] for key, value in manifest.items()],
    )

    omega = np.linspace(0.0, 8.0, 121)
    delta = np.empty_like(omega)
    delta[1:-1] = 0.5 * (omega[2:] - omega[:-2])
    delta[0] = 0.5 * (omega[1] - omega[0])
    delta[-1] = 0.5 * (omega[-1] - omega[-2])
    spectrum = np.exp(-0.5 * ((omega - 2.0) / 0.55) ** 2)
    spectrum /= np.sum(spectrum * delta)
    if basis == "bosonic_matsubara":
        indices = np.arange(point_count)
        coordinates = 2.0 * np.pi * indices / beta
    else:
        indices = np.arange(point_count)
        coordinates = np.linspace(0.0, 0.5, point_count)
    kernel = regularized_bosonic_kernel(basis, beta, coordinates, omega)
    means = kernel @ (spectrum * delta)
    covariance = np.diag(np.linspace(2.0e-5, 5.0e-5, point_count))
    errors = np.sqrt(np.diag(covariance))

    if basis == "bosonic_matsubara":
        header = [
            "momentum",
            "k_0",
            "frequency",
            "n",
            "omega",
            "mean",
            "standard_error",
            "exact_constraint",
        ]
        rows = [
            [0, 1, point, indices[point], coordinates[point], means[point], errors[point], 0]
            for point in range(point_count)
        ]
    else:
        header = [
            "momentum",
            "k_0",
            "lag",
            "tau",
            "mean",
            "standard_error",
            "exact_constraint",
        ]
        rows = [
            [0, 1, point, coordinates[point], means[point], errors[point], 0]
            for point in range(point_count)
        ]
    _write_tsv(path / "values.tsv", header, rows)
    _write_tsv(
        path / "covariance.tsv",
        ["momentum", "left", "right", "covariance_of_mean"],
        [
            [0, left, right, covariance[left, right]]
            for left in range(point_count)
            for right in range(point_count)
        ],
    )
    _write_tsv(
        path / "blocks.tsv",
        ["block", "momentum", "frequency_or_lag", "value"],
        [[block, 0, point, means[point]] for block in range(2) for point in range(point_count)],
    )
    return path


def test_regularized_kernels_have_finite_origin_limits():
    omega = np.array([0.0, 0.5, 2.0])
    matsubara = regularized_bosonic_kernel(
        "bosonic_matsubara", 2.0, np.array([0.0, np.pi]), omega
    )
    assert matsubara[0, 0] == 2.0
    assert matsubara[1, 0] == 0.0
    lag = regularized_bosonic_kernel(
        "imaginary_time_lag", 2.0, np.array([0.0, 0.5]), omega
    )
    assert np.all(np.isfinite(lag))
    assert np.all(lag[:, 0] == 1.0)


@pytest.mark.parametrize("basis", ["bosonic_matsubara", "imaginary_time_lag"])
def test_loads_both_bundle_bases_and_prepares_covariance(tmp_path, basis):
    bundle = _synthetic_bundle(tmp_path / basis, basis)
    data = load_density_continuation_bundle(bundle)
    options = MaxEntRunOptions(omega_max=8.0, omega_points=51, quiet=True)
    prepared = prepare_maxent_input(data, options)
    assert data.basis == basis
    assert data.momentum_indices == (1,)
    assert prepared.kernel.shape == (6, 51)
    assert prepared.transformed_kernel.shape == (6, 51)
    assert prepared.retained_covariance_rank == 6
    assert prepared.default_model_normalization > 0.0


def test_rejects_exact_zero_momentum_constraint(tmp_path):
    bundle = _synthetic_bundle(tmp_path / "bundle")
    values = list(csv.reader((bundle / "values.tsv").open(), delimiter="\t"))
    for row in values[1:]:
        row[1] = "0"
        row[-1] = "1"
    _write_tsv(bundle / "values.tsv", values[0], values[1:])
    with pytest.raises(BundleFormatError, match="exact constraints"):
        load_density_continuation_bundle(bundle)


def test_runs_vendored_data_kernel_and_writes_reconstruction(tmp_path):
    bundle = _synthetic_bundle(tmp_path / "bundle")
    data = load_density_continuation_bundle(bundle)
    options = MaxEntRunOptions(
        omega_max=8.0,
        omega_points=60,
        alpha_min=1.0e-2,
        alpha_max=1.0e2,
        alpha_points=8,
        max_iterations=500,
        quiet=True,
    )
    prepared = prepare_maxent_input(data, options)
    result = run_maxent(data, prepared, options)
    assert result.rescaled_spectrum.shape == (60,)
    assert np.all(np.isfinite(result.rescaled_spectrum))
    assert np.all(result.rescaled_spectrum >= 0.0)
    assert np.allclose(
        result.density_spectrum, result.omega * result.rescaled_spectrum
    )
    assert result.covariance_chi2 >= 0.0

    destination = tmp_path / "result"
    write_adapter_output(destination, data, prepared, options, result)
    assert (destination / "spectrum.tsv").is_file()
    assert (destination / "reconstruction.tsv").is_file()
    metadata = json.loads((destination / "run.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["triqs_maxent_version"] == "4.0.0"
    assert metadata["physical_spectrum"] == "A_density(omega)=omega*B(omega)"
