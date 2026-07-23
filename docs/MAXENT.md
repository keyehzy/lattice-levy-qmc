# Analytic continuation with vendored TRIQS/maxent

The repository vendors TRIQS/maxent 4.0.0 under `vendor/triqs_maxent` and
provides `python/qmc_maxent.py` to read the C++ interacting sampler's
`density-continuation-v1` bundles. The adapter supports both
`bosonic_matsubara` and `imaginary_time_lag` data, including each selected
momentum's dense covariance matrix.

TRIQS/maxent is GPL-3.0-or-later software. Its upstream source, notices, and
license files are retained verbatim; see `vendor/README.md`.

## Why a custom kernel is required

TRIQS/maxent's `TauMaxEnt` convenience class uses a fermionic Green-function
kernel. Density response has the positive bosonic convention

\[
\chi(q,i\omega_n)=\int_0^\infty d\omega\,
\frac{2\omega}{\omega^2+\omega_n^2}A_q(\omega)
\]

or

\[
C(q,\tau)=\int_0^\infty d\omega\,
\frac{\cosh[(\beta/2-\tau)\omega]}{\sinh(\beta\omega/2)}
A_q(\omega).
\]

The adapter therefore uses the upstream `DataKernel` and `MaxEntLoop`
directly. It continues the regularized nonnegative spectrum

\[
B_q(\omega)=A_q(\omega)/\omega,
\]

using kernels

\[
K_n(\omega)=\frac{2\omega^2}{\omega^2+\omega_n^2}
\]

and

\[
K_\tau(\omega)=
\omega\frac{\cosh[(\beta/2-\tau)\omega]}{\sinh(\beta\omega/2)}.
\]

These have finite limits at the origin. `spectrum.tsv` reports both
`rescaled_spectrum_B` and the requested physical
`density_spectral_function_A = omega * B`. A separate elastic delta peak at
zero frequency is outside this continuum parameterization.

## Run

First generate a continuation bundle with the interacting C++ demo:

```sh
cd cpp
cmake --preset dev
cmake --build --preset dev
./build/dev/examples/qmc_interacting_demo \
  --particles 6 --beta 1.5 --linear-size 8 --dimension 1 \
  --hopping 1.0 --interaction 2.0 --burn-in 500 --samples 3000 \
  --density-momenta 1 --density-frequency-max 8 \
  --density-measurements-per-block 30 \
  --density-continuation-dir ../density-continuation-v1 --no-trace
cd ..
```

Then choose an explicit real-frequency cutoff and run MaxEnt:

```sh
python3 python/qmc_maxent.py density-continuation-v1 \
  --momentum-ordinal 0 \
  --omega-max 12 \
  --omega-points 200 \
  --output-dir density-maxent-v1
```

`--omega-max`, the real-frequency mesh, alpha range, analyzer, and covariance
cutoff are analysis choices. They should be varied before assigning physical
significance to spectral features. Run `python3 python/qmc_maxent.py --help`
for all controls.

Use `--prepare-only` to validate and convert the bundle without performing the
optimization. This produces the exact arrays supplied to the upstream
`DataKernel`:

- `input_data.tsv`: selected mean and standard error;
- `covariance.txt`: original dense covariance matrix;
- `omega.txt`: positive real-frequency grid;
- `kernel.txt`: unrotated bosonic kernel;
- `transformed_data.tsv`: covariance eigenmodes retained by the cutoff; and
- `transformed_kernel.txt`: the kernel rotated into the same eigenbasis.

A full run additionally writes:

- `spectrum.tsv`: \(B(\omega)\), physical \(A(\omega)\), and the default model;
- `reconstruction.tsv`: observed and reconstructed imaginary-axis data;
- `alpha_diagnostics.tsv`: alpha, chi-squared, entropy, and cost; and
- `run.json`: complete adapter settings, upstream revision, selected analyzer
  result, covariance eigenvalues, and retained rank.

The adapter diagonalizes the covariance and drops modes whose eigenvalues are
at most `largest_eigenvalue * covariance_rcond`. Every dropped mode is recorded
in `run.json`; no ridge or diagonal replacement is applied. Generate more
blocks than input points when possible, and repeat the QMC blocking analysis
until errors are stable.

## Installation boundary

The array-based adapter uses NumPy, SciPy, and the vendored pure-Python MaxEnt
core. It does not require a system TRIQS installation. Building or importing
the full upstream package normally still requires TRIQS 4.0; this adapter does
not emulate TRIQS Green-function, HDF5, or MPI APIs.
