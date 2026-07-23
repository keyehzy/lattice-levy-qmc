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

## Batch dynamic structure factor

`python/qmc_dynamic_structure.py` runs the adapter for every measured nonzero
momentum in one bundle and converts the per-site dissipative spectrum to the
conventional positive-frequency, per-particle dynamic structure factor

\[
S(\mathbf q,\omega)
=\frac{V}{N}\frac{A_{\mathbf q}(\omega)}
 {1-\exp(-\beta\omega)},\qquad \omega>0.
\]

Because the adapter continues \(B_{\mathbf q}(\omega)=A_{\mathbf
q}(\omega)/\omega\), the regular mesh-origin value is

\[
S(\mathbf q,0)=\frac{V}{N\beta}B_{\mathbf q}(0).
\]

The C++ run must request every desired momentum row. For example, a complete
one-dimensional \(L=8\) lattice grid is
`--density-momenta '0;1;2;3;4;5;6;7'`. The zero row is useful as an exact
normalization diagnostic in the bundle but is skipped by batch continuation.

Run all nonzero momenta and propagate the block jackknife through MaxEnt with:

```sh
python3 python/qmc_dynamic_structure.py density-continuation-v1 \
  --omega-max 12 \
  --omega-points 200 \
  --output-dir dynamic-structure-v1
```

Use `--momentum-ordinals 1,3,5` to select bundle momentum rows explicitly.
The default skips the exact fixed-\(N\), zero-momentum constraint and processes
every other row. `--no-jackknife` performs only the full-sample reconstruction
when iterating on the real-frequency mesh or other MaxEnt controls.

The batch result contains:

- `dynamic_structure_factor.tsv`: combined \(S(\mathbf q,\omega)\), jackknife
  standard error, \(A_{\mathbf q}\), and \(B_{\mathbf q}\);
- `momentum_summary.tsv`: continuum-integrated static weight, peak frequency,
  their jackknife errors, selected alpha, covariance chi-squared, and retained
  covariance rank;
- `dynamic_structure_factor_map.png`: the \(q\)-\(\omega\) intensity map;
- `dynamic_structure_factor_linecuts.png`: selected momentum line cuts with
  jackknife error bands;
- `peak_dispersion.png`: the maximum-intensity frequency versus momentum;
- `momentum-NNNN/`: the original MaxEnt result plus per-momentum dynamic and
  jackknife tables; and
- `batch_run.json`: normalization, wavevector, MaxEnt, and jackknife
  provenance.

For one-dimensional bundles the plots use the centered first Brillouin zone.
For higher-dimensional bundles, every wavevector component is retained in the
tables and plots use the momentum-row axis because a general list of vectors
does not define a unique path through momentum space.

For every leave-one-block-out reconstruction, the adapter holds fixed the
covariance matrix and retained eigenspace, real-frequency mesh, default model,
alpha mesh, and analyzer. Only the jackknife mean changes. The analyzer still
selects its result from the same alpha mesh for each replicate, and the
selected alpha is recorded. `jackknife_spectra.tsv` retains every reconstructed
replicate rather than only the final standard error.

The reported continuum static sum is

\[
S(\mathbf q)=\frac{V}{N}\int_0^\infty d\omega\,
A_{\mathbf q}(\omega)\coth(\beta\omega/2).
\]

It excludes the same elastic zero-frequency delta peak as the MaxEnt
parameterization. Peak positions are real-frequency-mesh observables and their
uncertainty cannot be smaller than the chosen mesh spacing without a separate
peak-interpolation model.

## Installation boundary

The array-based adapter uses NumPy, SciPy, and the vendored pure-Python MaxEnt
core. It does not require a system TRIQS installation. Building or importing
the full upstream package normally still requires TRIQS 4.0; this adapter does
not emulate TRIQS Green-function, HDF5, or MPI APIs.
