# Analytic-continuation data from continuous-time density measurements

Date: 2026-07-23

Status: staged implementation specification. The exact continuous-time
Matsubara density estimator, its block-resolved statistics, and the versioned
continuation-data export/demo workflow are implemented. Direct requested-lag
output is not implemented.

## Goal and boundary

The interacting sampler should be able to produce a statistically qualified
imaginary-axis density correlation that can be consumed by MaxEnt, stochastic
analytic continuation, or another external continuation program. The library
should not initially implement an analytic-continuation solver or select a
default model, real-frequency grid, regularisation strength, or spectral
parameterisation.

The first supported input basis should be the already implemented bosonic
Matsubara susceptibility. A second exact backend should provide selected
imaginary-time lags when a continuation workflow specifically needs
`C(q,tau)`. Both paths must preserve the continuous event representation; a
requested lag is an evaluation point, not a retained measurement grid or a
Trotter discretisation.

The deliverable is therefore:

1. block-level density observations from an interacting Markov chain;
2. means, standard errors, and cross-frequency **statistical** covariance;
3. leave-one-block-out data for jackknife propagation through a nonlinear
   continuation;
4. a versioned, self-describing export bundle; and
5. optionally, exact selected-lag density correlations with the same
   statistical and export conventions.

The continuation itself remains downstream analysis. This boundary keeps the
QMC library independent of a particular ill-posed inverse method while
providing the data and error model those methods require. MaxEnt
implementations can consume either imaginary-time or Matsubara-frequency data,
and correlated input components enter their likelihood through the covariance
matrix; see Levy, LeBlanc, and Gull,
[Implementation of the Maximum Entropy Method for Analytic Continuation](https://arxiv.org/abs/1606.00368),
and Jarrell and Gubernatis,
[Bayesian inference and the analytic continuation of imaginary-time quantum Monte Carlo data](https://doi.org/10.1016/0370-1573(95)00074-7).
Stochastic analytic continuation is an equally valid consumer; see Shao and
Sandvik,
[Progress on stochastic analytic continuation of quantum Monte Carlo data](https://doi.org/10.1016/j.physrep.2022.11.002).

## Existing measurement and missing workflow

For a selected lattice momentum `q`, the current continuous projector returns
the unnormalised per-configuration amplitude

\[
\rho_{\mathbf qn}
=\int_0^\beta d\tau\,
e^{i\omega_n\tau} n_{\mathbf q}(\tau),\qquad
n_{\mathbf q}(\tau)
=\sum_a e^{-i\mathbf q\cdot\mathbf x_a(\tau)}.
\]

`DensityMatsubaraAccumulator` then returns the connected susceptibility

\[
\chi_{nn}(\mathbf q,i\omega_n)
=\frac{1}{\beta V}
 \left\langle
 \left|\rho_{\mathbf qn}
       -\beta N\delta_{\mathbf q,0}\delta_{n,0}\right|^2
 \right\rangle.
\]

This is already the correct imaginary-axis density response for the
homogeneous fixed-particle-number model. Interaction `U` enters through the
probability of the sampled `ContinuousConfiguration`; the projector itself
needs only the configuration's free-model geometry. The finite-interaction
statistical regression compares this estimator with exact diagonalisation.

The current accumulator deliberately retains only its running sums and sample
count. It does not preserve the sequence of Markov-chain observations or
estimate their autocorrelation. Consequently it cannot recover block means or
cross-frequency sampling covariance after `finish()`. The interacting demo
also writes only scalar observables. These are workflow gaps, not defects in
the density estimator.

## Observable and spectral convention

Define

\[
\delta n_{\mathbf q}(\tau)
=n_{\mathbf q}(\tau)-N\delta_{\mathbf q,0}.
\]

The imaginary-time density correlation in the existing per-site convention is

\[
C_{nn}(\mathbf q,\tau)
=\frac{1}{V}
 \left\langle
 \delta n_{\mathbf q}(\tau)\delta n_{-\mathbf q}(0)
 \right\rangle,\qquad 0\le\tau<\beta.
\]

Its continuous time-origin estimator on one configuration is

\[
c_{\mathbf q,\tau}
=\frac{1}{\beta V}\int_0^\beta ds\,
 \delta n_{\mathbf q}(s+\tau)
 \delta n_{-\mathbf q}(s),
\]

where `s + tau` is reduced periodically modulo `beta`. Equilibrium time
translation makes its ensemble mean equal to `C_nn(q,tau)`. Its bosonic
Matsubara transform is

\[
\int_0^\beta d\tau\,
e^{i\omega_n\tau} C_{nn}(\mathbf q,\tau)
=\chi_{nn}(\mathbf q,i\omega_n),
\]

which fixes the sign and normalisation shared by the two backends.

For a positive dissipative density spectral function

\[
A_{\mathbf q}(\omega)
=-\frac{1}{\pi}\operatorname{Im}
  \chi^R_{nn}(\mathbf q,\omega),\qquad \omega>0,
\]

the corresponding kernels are

\[
\chi_{nn}(\mathbf q,i\omega_n)
=\int_0^\infty d\omega\,
\frac{2\omega}{\omega^2+\omega_n^2}A_{\mathbf q}(\omega)
\]

and

\[
C_{nn}(\mathbf q,\tau)
=\int_0^\infty d\omega\,
\frac{\cosh[(\beta/2-\tau)\omega]}
     {\sinh(\beta\omega/2)}
A_{\mathbf q}(\omega),
\]

apart from any explicitly modelled zero-frequency elastic contribution. The
export must identify this convention and must not label the imaginary-axis
values themselves as a dynamic structure factor or spectral function.
Continuation programs sometimes rescale the bosonic spectrum to remove the
kernel's small-frequency singular behaviour; that rescaling belongs to the
consumer and must be recorded in its analysis.

Only nonnegative Matsubara indices are needed for this real density
auto-correlation. Negative indices remain supported by the lower-level
projector for identity tests and other observables. At fixed particle number,
every connected `q == 0` value is exactly zero and contains no continuation
information; a continuation export should reject a `q == 0`-only request or
mark those rows as exact constraints rather than noisy data.

## Statistical observation and blocking

For mode `m = (q,n)`, define the real per-configuration observation

\[
g_{i,m}
=\frac{1}{\beta V}
 \left|\rho_{i,m}
       -\beta N\delta_{\mathbf q,0}\delta_{n,0}\right|^2.
\]

These values are generally correlated both along Monte Carlo time and across
requested frequencies. Thinning does not by itself establish independence.
The first analysis type should therefore form consecutive, equal-size blocks
after burn-in:

\[
b_{j,m}=\frac{1}{s}\sum_{i=js}^{(j+1)s-1}g_{i,m},
\qquad j=0,\ldots,B-1,
\]

where `s` is the explicitly recorded measurements-per-block. The initial
implementation should require a complete final block. It must not silently
drop or reweight a partial block; the run workflow should choose a measurement
count divisible by `s`, or explicitly discard the partial block before
finishing.

The reported mean and covariance of that mean are

\[
\bar g_m=\frac1B\sum_j b_{j,m},
\]

\[
\Sigma_{mm'}
=\frac{1}{B(B-1)}
 \sum_j (b_{j,m}-\bar g_m)(b_{j,m'}-\bar g_{m'}).
\]

The standard error is `sqrt(Sigma_mm)`. This formula treats sufficiently long
block means as the approximately independent estimates supplied to a
continuation likelihood.

This covariance is not the physical cross-mode response discussed in
`CONTINUOUS_TIME_MEASUREMENTS.md`. That physical response would correlate
distinct Fourier components as observables and is diagonal by translation
invariance in the present model. `Sigma` instead describes correlated
**errors of the estimated diagonal susceptibilities**. It is generally
non-diagonal in frequency and is required even though the physical response is
mode-diagonal.

Continuation normally runs independently for each momentum. The initial
result should therefore expose one `F x F` frequency covariance matrix per
requested momentum, while retaining block values for all requested modes. It
should not allocate a `(QF) x (QF)` matrix unless a later joint-momentum
analysis requests one. Storing block values costs `O(BQF)`; producing all
per-momentum covariance matrices costs `O(QF^2)`.

An empirical covariance has rank at most `B - 1`. A result with `B <= F`
cannot provide an invertible full frequency covariance. Construction may
preserve such data, but export must report the rank condition and must not
silently diagonalise, shrink, truncate, or add a ridge. Any regularisation of
the covariance is a named downstream analysis choice.

### Block-size evidence

The library cannot infer a universally correct block size from one short run.
The workflow should accept an explicit positive block size and retain enough
block data for the caller to compare successively coarser blocking levels.
Documentation and the demo should recommend increasing `s` until estimated
standard errors and the leading covariance eigenspace are stable within their
own noise.

A later automatic diagnostic may estimate integrated autocorrelation times,
but it must report its windowing rule and uncertainty. It must not silently
change the production block size. The final bundle records burn-in, thinning,
block size, completed block count, and the total number of post-burn-in sampler
sweeps represented by one block.

### Jackknife

For each complete block, the leave-one-block-out mean is

\[
\bar g_m^{(-j)}
=\frac{B\bar g_m-b_{j,m}}{B-1}.
\]

For the linear imaginary-axis mean this contains no information beyond the
block series and covariance. Its purpose is to propagate sampling uncertainty
through a nonlinear continuation: run the same downstream continuation on
each `bar(g)^(-j)` dataset, then form jackknife errors for spectral summaries.

The QMC library should expose or export the replicates but should not claim
that a jackknife error on the imaginary-axis mean quantifies MaxEnt
regularisation or default-model uncertainty. A continuation consumer is
responsible for holding all non-data choices fixed across replicates and for
recording them.

## Implemented block-statistics ownership

The core `DensityMatsubaraAccumulator` remains the small ensemble-mean
accumulator. A separate analysis accumulator owns block state:

```cpp
class DensityMatsubaraBlockSeries {
public:
  [[nodiscard]] const Model &model() const noexcept;
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  [[nodiscard]] std::size_t measurements_per_block() const noexcept;
  [[nodiscard]] std::size_t block_count() const noexcept;
  [[nodiscard]] std::size_t sample_count() const noexcept;

  [[nodiscard]] double block_value(
      std::size_t block, std::size_t frequency,
      std::size_t momentum) const;
  [[nodiscard]] double mean(
      std::size_t frequency, std::size_t momentum) const;
  [[nodiscard]] double covariance_of_mean(
      std::size_t momentum, std::size_t left_frequency,
      std::size_t right_frequency) const;
  [[nodiscard]] double standard_error(
      std::size_t frequency, std::size_t momentum) const;
  [[nodiscard]] double jackknife_mean(
      std::size_t omitted_block, std::size_t frequency,
      std::size_t momentum) const;
};

class DensityMatsubaraBlockAccumulator {
public:
  DensityMatsubaraBlockAccumulator(
      Model, MatsubaraModeSet, std::size_t measurements_per_block);

  [[nodiscard]] const Model &model() const noexcept;
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  [[nodiscard]] std::size_t measurements_per_block() const noexcept;
  [[nodiscard]] std::size_t completed_block_count() const noexcept;
  [[nodiscard]] std::size_t pending_sample_count() const noexcept;

  void observe(const ContinuousParticleModes &);
  [[nodiscard]] DensityMatsubaraBlockSeries finish() const;
};
```

These names and ownership rules are implemented:

- `observe()` consumes the same primitive values as the existing accumulator;
- analytic centring is identical to `DensityMatsubaraAccumulator`;
- block values are normalised `g_(i,m)`, not raw amplitudes;
- every mode/model/finite check occurs before mutation;
- `finish()` requires at least two complete blocks and no pending partial
  block;
- all block, covariance, and jackknife extents are checked before allocation;
- the finished series owns its model, mode set, block size, values, and exact
  sample count; and
- accessors check block, momentum, and frequency indices.

The implementation should share a private function for the per-configuration
centred density observation so the simple and block accumulators cannot drift
in normalization. Given identical observations, the block-series mean must
equal `DensityMatsubaraAccumulator::finish().at(...)` to roundoff.

`DensityMatsubaraBlockSeries` should calculate covariance with a stable
two-pass or online pairwise algorithm over completed block means. It may cache
the result at construction, but it must not lazily mutate through a `const`
accessor. Because `g` is real, the sampling covariance is a real symmetric
matrix; calculate one triangle and mirror it structurally.

## Implemented versioned continuation-data bundle

The initial interchange format is a directory of small UTF-8, tab-separated
files rather than a solver-specific input file or a new binary dependency:

```text
density-continuation-v1/
  manifest.tsv
  values.tsv
  covariance.tsv
  blocks.tsv
```

`manifest.tsv` contains one key/value pair per row and records:

- schema identifier and version;
- complete `InteractingModel`: particle count, `beta`, linear size,
  dimension, hopping, and `U`;
- lattice volume and units convention;
- seed;
- burn-in sweeps, thinning sweeps, and the complete sweep/update plan;
- measurements per block, completed block count, and sample count;
- Fourier signs and normalization identifier;
- basis: `bosonic_matsubara` or `imaginary_time_lag`;
- observable identifier: `connected_density_per_site`;
- kernel convention identifier;
- whether the rows are measured, exact constraints, or diagnostics; and
- the command/program version that produced the bundle.

`values.tsv` contains one row per requested `(q,n)` or `(q,tau)` with:

- momentum ordinal and every integer momentum component;
- frequency/lag ordinal;
- signed Matsubara index and derived physical frequency, or lag in the
  repository's inverse-energy units;
- mean;
- standard error; and
- an exact-constraint flag.

`covariance.tsv` stores one dense statistical frequency/lag covariance matrix
per momentum in coordinate form:

```text
momentum  left  right  covariance_of_mean
```

`blocks.tsv` stores the normalized block observations:

```text
block  momentum  frequency_or_lag  value
```

Keeping block data makes the covariance reproducible, permits coarser
reblocking without rerunning QMC, and supplies the leave-one-block-out
replicates. A solver-specific adapter may translate this bundle to a MaxEnt or
stochastic-continuation input format, but the QMC demo should not emit a format
that implies one downstream implementation is authoritative.

The writer validates all tables and metadata before creating the destination.
It writes to a sibling temporary directory and atomically renames the complete
bundle where the platform supports it. It rejects an existing destination and
reports partial cleanup failures. The private writer support code belongs to
the example/run workflow rather than the core measurement types.

The complete `InteractingModel` is run provenance even though primitive
particle modes own only the free `Model`. A bundle must never infer `U` from a
measurement result; the run workflow supplies it explicitly and verifies that
its free model matches the block series.

## Direct requested-lag backend

Some continuation programs and diagnostic plots prefer `C(q,tau)` to
Matsubara data. Inverting a finite set of Matsubara modes is not an exact
continuous-time estimator and can ring at the frequency cutoff. The direct
backend should therefore evaluate selected lags from residence intervals.

Conceptually:

```cpp
struct ImaginaryTimeLagRequest {
  std::vector<std::vector<std::size_t>> momentum_indices;
  std::vector<double> lags;
};

class ImaginaryTimeLagSet {
public:
  ImaginaryTimeLagSet(
      double beta, TorusLayout, ImaginaryTimeLagRequest);

  [[nodiscard]] double beta() const noexcept;
  [[nodiscard]] const TorusLayout &layout() const noexcept;
  [[nodiscard]] std::size_t momentum_count() const noexcept;
  [[nodiscard]] std::size_t lag_count() const noexcept;
  [[nodiscard]] double lag(std::size_t) const;
};

class ContinuousDensityLagPlan {
public:
  explicit ContinuousDensityLagPlan(ImaginaryTimeLagSet);
  [[nodiscard]] const ImaginaryTimeLagSet &lags() const noexcept;
};

class ContinuousDensityLagValues {
public:
  [[nodiscard]] const Model &model() const noexcept;
  [[nodiscard]] const ImaginaryTimeLagSet &lags() const noexcept;
  // Integral ds Re[n_q(s+lag) n_-q(s)].
  [[nodiscard]] double overlap(
      std::size_t lag, std::size_t momentum) const;
};

[[nodiscard]] ContinuousDensityLagValues
continuous_density_lag_values(
    const ContinuousMeasurementContext &,
    const ContinuousDensityLagPlan &);
```

The exact public names can change during implementation. The required
semantics are:

- `beta` is positive and every lag is finite and canonical in `[0,beta)`;
- momentum and lag requests are nonempty, duplicate-free, and preserve request
  order;
- the result owns complete free-model and lag/momentum provenance;
- the per-configuration value is an unnormalised, uncentred time overlap,
  analogous to the unnormalised amplitudes in `ContinuousParticleModes`;
- the source-free auto-correlation stores the real, time-reversal-symmetrised
  estimator, eliminating an exactly zero ensemble imaginary part without
  changing its mean;
- the accumulator subtracts `beta*N*N` only at `q == 0`, applies
  `1/(beta*V)`, owns sample/block counts, and installs the exact fixed-`N`
  zero rather than subtracting two rounded values; and
- a future background gauge field or genuinely complex mixed correlation
  requires a different result type rather than silently discarding its
  imaginary part.

### Exact interval-overlap algorithm

Replay the shared event groups once to construct the piecewise-constant
`n_q(s)` residence intervals for every requested momentum. For one lag,
cyclically shift a second view of those intervals by the lag and intersect the
two periodic partitions with a two-pointer sweep. Each intersection contributes

\[
\text{overlap length}\times
\operatorname{Re}\left[
n_{\mathbf q}(s+\tau)
n_{\mathbf q}(s)^*
\right].
\]

Split a shifted interval only when it crosses the periodic seam. Event groups
at zero are applied before the first positive-duration interval; events at
`beta` close the seam after the final interval, exactly as in the Matsubara
projector. Isolated boundary values have zero residence measure, but using the
shared context prevents a second, inconsistent seam convention.

For `G` event groups, `Q` momenta, and `T` requested lags, the direct baseline
is `O(QG + QTG)` time and `O(QG + QT)` working/result storage. Do not introduce
a uniform time grid. Dense-lag convolution or FFT acceleration requires a
benchmark against this exact interval baseline.

The initial output is selected momentum rather than all real-space
displacements because continuation is normally performed independently at
fixed `q`. A complete momentum request can be inverse-transformed spatially to
`C(r,tau)` without approximating time. A direct selected-displacement backend
should wait for a concrete real-space consumer.

## Sampling and demo workflow

Measurement orchestration remains outside `InteractingSampler`. Until the
streaming-run work tracked separately by repository issue 5 is implemented, a
caller can construct the plan and block accumulator around the existing
explicit sampling loop.

The interacting demo has opt-in density-continuation arguments for:

- `--density-momenta`, using semicolon-separated rows and comma-separated
  components;
- `--density-frequency-max`, selecting the inclusive nonnegative range from
  zero through the supplied Matsubara index;
- `--density-measurements-per-block`;
- `--density-continuation-dir`; and
- `--no-trace`, when the existing scalar trace should not be retained beside
  the bundle.

The defaults preserve the current inexpensive scalar demo. Continuation
request validation and all output-path checks occur before sampler
construction. The requested measurement count must form at least two complete
blocks. The demo supplies the complete interacting model and run schedule to
the bundle writer and prints the number of completed blocks plus the largest
standard error.

When a general streaming observer becomes available, the demo should adopt it
without changing the block-series or bundle formats. The measurement observer
must see accepted configurations in the same order as the current explicit
loop.

## Verification plan

### Deterministic statistics and shape tests

- A hand-authored block table reproduces its mean, covariance of the mean,
  standard errors, and every leave-one-block-out mean.
- Covariance is structurally symmetric, its diagonal is nonnegative, and its
  rank warning is reported when `B <= F`.
- Two identical-frequency observation streams with different requested order
  produce correspondingly permuted means and covariance.
- Block and simple accumulators return the same mean on identical primitive
  observations.
- Model/mode mismatch, nonfinite observations, extent overflow, incomplete
  final blocks, fewer than two blocks, and count overflow fail before
  mutation/publication.
- Every result accessor checks block, frequency/lag, momentum, and covariance
  axes.
- Exported means and covariance can be recomputed exactly to roundoff from
  `blocks.tsv`; every manifest count agrees with the tables.
- Writer failure leaves no published partial bundle and never overwrites an
  existing destination by default.

### Requested-lag identities

- The raw `q == 0` overlap is exactly `beta*N*N`, and every accumulated
  connected value is exactly zero.
- Lag zero equals the time-origin-averaged equal-time density norm and is
  nonnegative configuration by configuration.
- The symmetrised value satisfies `C(q,tau) == C(q,beta-tau)` to a tight
  deterministic tolerance.
- Global covering translation and time-origin rotation leave the lag
  auto-correlation unchanged.
- Static, single-event, coincident-event, zero/beta seam, single-site, empty,
  and multiple-cycle configurations match a direct interval-intersection
  reference.
- Exact integration of the piecewise lag correlation against a Matsubara phase
  reproduces `|delta rho(q,n)|^2/(beta*V)` for deterministic fixtures. Sampling
  a finite lag list and applying a quadrature is explicitly not this test.
- Small-system Lehmann values agree at several lags and momenta.

### Markov-chain and continuation-data tests

- The existing finite-`U` small-system run is extended to write block data and
  reproduce its blocked standard errors through the public analysis type.
- Increasing the block size reaches a stable error plateau on a documented
  seeded workload; the test tolerance is derived from the number of blocks and
  remains in the labelled statistical target.
- Matsubara means remain consistent with exact diagonalisation while their
  exported covariance is finite and positive semidefinite to numerical
  tolerance.
- A deterministic synthetic positive spectrum is forward-evaluated with the
  documented kernel and the resulting imaginary-axis table round-trips through
  the bundle reader used by a test adapter. Recovering that spectrum with
  MaxEnt is an analysis demonstration, not a pass/fail library test.

## Implementation order

1. **Completed 2026-07-23:** Add `DensityMatsubaraBlockAccumulator` and its
   provenance-owning block-series result over the existing primitive particle
   modes.
2. **Completed 2026-07-23:** Add per-momentum covariance, standard-error, and
   leave-one-block-out accessors with deterministic table tests.
3. **Completed 2026-07-23:** Add the versioned bundle writer and opt-in
   interacting-demo workflow, including complete run and interaction
   provenance.
4. Extend the finite-`U` statistical regression to exercise the public block
   workflow and exported values.
5. Add the separate requested-lag plan, exact interval-overlap projector, block
   accumulator, and deterministic/Lehmann tests when a direct-`tau` consumer is
   selected.
6. Provide thin, separately maintained adapters for concrete continuation
   programs only after their input conventions are pinned by integration
   tests. Do not add a continuation solver to the QMC core by default.

## Non-goals

- No Trotter, retained, or quadrature time grid for interacting measurements.
- No inverse transform of a truncated Matsubara set presented as exact
  `C(q,tau)`.
- No automatic MaxEnt default model, real-frequency cutoff/grid,
  regularisation selection, or covariance regularisation.
- No claim that jackknife variation captures continuation-method systematic
  uncertainty.
- No continuation of `q == 0` fixed-`N` density data.
- No conductivity continuation from the hopping response; its contact terms,
  units, charge factors, and order of limits require a separate design.
- No full cross-momentum statistical covariance without a concrete joint
  continuation workflow.
