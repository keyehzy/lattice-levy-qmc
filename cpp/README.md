# C++ canonical lattice Lévy QMC

This directory contains C++20 implementations of both samplers. The
`qmc::ideal` target provides independent ideal-gas samples and retained-grid
measurements. The `qmc::interacting` target layers sparse continuous-time paths,
the exact on-site interaction action, Metropolis updates, and finite-`U`
observables on the same free numerical foundation.

The interacting representation stores every nearest-neighbor hopping event and
has no imaginary-time grid or Trotter approximation. The ideal `M` parameter is
only the number of bridge observations retained for measurement.

The C++ project follows [Semantic Versioning](https://semver.org/). While the
public API is below `1.0.0`, a minor release may contain breaking API changes;
patch releases contain backward-compatible fixes. `project(VERSION ...)` in
`CMakeLists.txt` is the source of truth, and the generated `<qmc/version.hpp>`
header exposes `qmc::kVersion` and its numeric components at compile time. See
the [changelog](CHANGELOG.md) for release history and pending changes.

The build requires CMake 3.24 or newer, a C++20 compiler, Ninja, and GSL 2.7
or newer. Examples use an installed cxxopts 3.3 or newer when available and
otherwise download a pinned upstream revision. Tests similarly use an installed
GoogleTest 1.17 when available and otherwise download its pinned upstream
revision.

## Build and test

Configure, build, and run the fast deterministic/invariant suite from this
directory:

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

Formatting and static analysis are available through:

```sh
cmake --build --preset dev --target format-check
cmake --preset tidy
cmake --build --preset tidy
ctest --preset tidy
```

The tidy preset analyzes the library and example targets with the project
`.clang-tidy` profile. GoogleTest translation units still compile in this
preset, but are excluded from clang-tidy because macro expansion makes their
diagnostics disproportionately noisy.

Run the distribution-level free and finite-`U` exact-diagonalization tests with:

```sh
cmake --preset statistical
cmake --build --preset statistical
ctest --preset statistical
```

AddressSanitizer and UndefinedBehaviorSanitizer are configured separately:

```sh
cmake --preset sanitizers
cmake --build --preset sanitizers
ctest --preset sanitizers
```

Release builds do not download test dependencies. When examples are enabled,
cxxopts may still be downloaded if it is not installed:

```sh
cmake --preset release
cmake --build --preset release
```

## Library use

Include the umbrella header and pass one explicit RNG through all stochastic
operations:

```cpp
#include <qmc/ideal.hpp>

qmc::Model model(qmc::ModelParameters{
    .particle_count = 8,
    .beta = 2.0,
    .linear_size = 16,
    .dimension = 2,
    .hopping = 1.0,
});
const qmc::CanonicalEnsemble ensemble(model);
qmc::Random random(2026);
const auto configuration =
    qmc::sample_ideal_boson_configuration(ensemble, 64, random);
configuration.validate();
const auto thermodynamics = qmc::canonical_thermodynamics(ensemble);
const auto momentum = qmc::momentum_distribution(ensemble);
const qmc::RetainedMeasurementContext measurements(configuration);
const auto equal_time = qmc::equal_time_observables(measurements);
const auto correlations = qmc::retained_density_correlations(measurements);
```

`CanonicalEnsemble` owns the validated model and its matching canonical
recursion together with immutable `OneParticleSpectrum` and `FreePathKernels`
values. The spectrum owns the checked torus layout and `O(L)` one-dimensional
momentum, sine, and cosine tables; full `L^d` momentum records are produced only
when requested. The path kernels bind hopping, numerical work limits, and torus
provenance once for retained and continuous bridge operations. Pass custom
`NumericalOptions` to the ensemble constructor when those limits should be
reused across samples. Retain the ensemble when drawing multiple samples or
evaluating several exact observables; model-only and scalar-heavy overloads
remain available for one-off calls. Its read-only recursion views and
cycle/statistics overloads may also reuse any particle-number prefix through
the ensemble's configured maximum.

`configuration.covering_worldlines()` exposes the retained unwrapped
coordinates through a read-only, shape-safe buffer;
`configuration.topology()` exposes the matching validated permutation and its
cycle views. Torus coordinates and cycle windings are derived from those two
authoritative values instead of being stored as synchronized copies. `64` is
the number of retained time links per interval `beta`; it is an observation
resolution, not a Trotter discretization.

`RetainedMeasurementContext` owns the retained-grid provenance and physical
site positions derived from a configuration. Reuse one when evaluating several
retained observables for the same sample. Configuration-taking convenience
overloads remain available for one-off measurements.

`qmc::TorusLayout` is the shared checked geometry for flattened lattice data.
It fixes axis zero as the least-significant base-`L` digit and produces compact
`qmc::SiteId` values for physical sites, while covering-space coordinates remain
separate so winding is never discarded.

For the interacting chain, include the separate umbrella header:

```cpp
#include <qmc/interacting.hpp>

qmc::InteractingModel model{
    .free = qmc::Model(qmc::ModelParameters{
        .particle_count = 6,
        .beta = 1.5,
        .linear_size = 8,
        .dimension = 1,
        .hopping = 1.0,
    }),
    .interaction = 2.0,
};
qmc::InteractingSampler sampler(model, 20260717);
qmc::SweepOptions sweep{
    .segment_updates = model.free.particle_count(),
    .segment_fraction = 0.35,
    .cycle_updates = 1,
    .global_updates = 1,
    .stitch_mixture = {},
};
qmc::RandomSeamStitchOptions stitches{
    .updates = model.free.particle_count(),
    .fraction = 0.75,
};
for (std::size_t index = 0; index < 500; ++index) {
  sampler.random_seam_stitch_sweep(stitches);
  sampler.sweep(sweep);
}
const auto samples = sampler.run(
    2'000, qmc::RunOptions{.burn_in = 0, .thin = 2, .sweep = sweep});
```

Segment moves redraw fixed-endpoint intervals and cycle moves redraw one
existing cycle including its winding. Random-seam stitch moves redraw `k<=8`
closed strands using a permanent-sampled endpoint matching and rearrange
permutation successors without opening a path;
the recommended `random_seam_stitch_sweep()` wraps fixed-seam attempts in two
uniform time-origin rotations to give the reversible `A B^m A` kernel. Global
ideal proposals remain available as a small-system cross-check. Interaction-
corrected moves use only the action difference in their Metropolis ratio.
`SegmentUpdateOptions`, `StitchUpdateOptions`, `StitchSweepOptions`, and
`RandomSeamStitchOptions` name move parameters and validate the complete request
before the sampler consumes randomness. Compound sweep and run plans likewise
validate once before their first update.
`state()` exposes the accepted configuration read-only, while `statistics()`
reports attempts, acceptances, stitch topology changes, and changed successors.
The configuration's `topology()` view provides validated successors and a
deterministic cycle decomposition from one authoritative value.
`measure_interaction(configuration, model)` returns action, pair overlap,
double occupancy, kinetic/interaction/total energy, and event count from one
independent full overlap sweep. Use it for arbitrary configurations and audit
checks; `sampler.observables()` derives the same scalar bundle from the
accepted-state overlap cache.

`ContinuousMeasurementContext` owns the physical positions immediately before
time-zero events and the configuration's globally ordered hopping geometry.
Equal-time hops are exposed as atomic groups, including distinct groups at
`0` and `beta`, so exact continuous-time projectors can share one boundary and
tie-order convention without retaining an imaginary-time grid.
`MatsubaraModeSet` separately owns selected torus momenta and signed frequency
indices in frequency-major order. `ContinuousMatsubaraPlan` validates the
continuous phase-accuracy bound and precomputes immutable spatial factors for
reuse across configurations. `continuous_particle_modes(context, plan)` then
projects exact density residence integrals and signed bond-midpoint hopping
impulses together in one grouped event sweep. Its unnormalised
`ContinuousParticleModes` result owns the complete free model and mode set,
and reports axis-resolved event counts for later response estimators.
`DensityMatsubaraAccumulator` binds that complete model and mode set, rejects
incompatible samples before mutation, and returns
`ContinuousMatsubaraDensityCorrelations`. The result reports the sampled
complex mean amplitude and the connected susceptibility
`<|delta rho(q,n)|^2>/(beta*V)` with the exact homogeneous fixed-particle-number
mean subtracted at amplitude level. `DensityMatsubaraBlockAccumulator` consumes
the same normalized observation into consecutive equal-size blocks; its owning
result exposes block values, means, per-momentum frequency covariance of the
mean, standard errors, and leave-one-block-out means.
`continuous_density_lag_values(context, plan)` separately projects exact
selected-lag density overlaps without a time grid. `DensityLagBlockAccumulator`
normalizes those overlaps into connected per-site correlations, installs the
fixed-particle-number zero-momentum value as exact zero, and publishes signed
block values, means, per-momentum cross-lag covariance, standard errors, and
leave-one-block-out means in an owning `DensityLagBlockSeries`.
`HoppingResponseAccumulator` consumes the same primitive sample and returns the
full signed-flux gauge response
`R=<I I^*>/(beta*V)`, axis-resolved diamagnetic term
`D=<K_axis>/(beta*V)`, and derived paramagnetic current correlation
`Lambda^p=D*delta-R`, together with sampled complex mean flux diagnostics.
`HoppingResponseBlockAccumulator` forms complete equal-size blocks of those
terms. Its owning series exposes authoritative block values, component-wise
standard errors, and leave-one-block-out means; paramagnetic statistics are
derived block by block so the covariance between `D` and `R` is retained.
The interacting demo's opt-in `--hopping-*` workflow writes a versioned
`hopping-response-v1` directory containing full run provenance and separate
response, diamagnetic, and mean-flux value/block tables.
`continuous_pair_density_modes(context, plan)` is the separate
occupancy-based diagonal projector. Its unnormalised
`ContinuousPairDensityModes` result contains the exact residence transform of
`sum_x exp(-i*q*x)*choose(n_x,2)`; coincident event groups are atomic, and its
zero-momentum, zero-frequency value exactly matches `pair_overlap_time`.

See [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md) for estimator definitions,
normalizations, exactness, and retained-grid conventions. The event-based
continuous measurement design is documented in
[`docs/CONTINUOUS_TIME_MEASUREMENTS.md`](../docs/CONTINUOUS_TIME_MEASUREMENTS.md).
The implemented Matsubara and requested-lag block statistics and continuation
data export workflows are specified in
[`docs/ANALYTIC_CONTINUATION_DATA.md`](../docs/ANALYTIC_CONTINUATION_DATA.md).

## Measurement and plotting demo

The example averages independent exact configurations, logs summary values,
writes full-dimensional data tables, and generates a gnuplot script. Pass
`--plot` to run gnuplot and render seven PNG summaries:

```sh
./build/dev/examples/qmc_ideal_demo \
  --particles 8 --beta 2.0 --time-links 64 --linear-size 16 \
  --dimension 2 --hopping 1.0 --seed 2026 --samples 1000 \
  --output-dir ideal_observables --plot
```

The output includes thermodynamics through `N`, momentum occupations and
fluctuations, the one-body density matrix, exact and sampled cycle statistics,
raw and conditioned winding/cycle geometry, equal-time density data, retained
imaginary-time correlations and Matsubara transforms, mean-square displacement,
return probabilities, and bridge displacement distributions. Plotting is
optional so the library and demo still run on systems without gnuplot; an
alternate executable can be supplied with `--gnuplot`.

Run it with `--help` to see parameter descriptions and defaults. The positional
form `[N beta M L d t seed]` remains supported for compatibility.

## Interacting trace demo

The continuous-time driver writes one row per measurement and prints energy,
double-occupancy, winding, and move-acceptance summaries:

```sh
./build/dev/examples/qmc_interacting_demo \
  --particles 6 --beta 1.5 --linear-size 8 --dimension 1 \
  --hopping 1.0 --interaction 2.0 --burn-in 500 --samples 3000 \
  --stitch-updates 6 --stitch-fraction 0.75 \
  --stitch-strand-counts 2,3,4 --stitch-strand-weights 0.8,0.15,0.05 \
  --output interacting_trace.dat
```

Random-seam stitching defaults to `max(1, N)` attempts per sweep and runs in a
hybrid schedule with `N` segment updates and one cycle update; the default
strand mixture remains pair-only and global ideal-gas proposals default to
zero. Use `--help` for scheduling, locality, collective strand mixtures,
thinning, seed, and output options. The derivation is in
[`docs/RANDOM_SEAM_STITCH.md`](../docs/RANDOM_SEAM_STITCH.md).

Continuation-ready density data are opt-in. This example measures the
nonzero one-dimensional momentum `k=1`, Matsubara indices `0` through `8`, and
writes four self-describing TSV tables without retaining the scalar trace:

```sh
./build/dev/examples/qmc_interacting_demo \
  --particles 6 --beta 1.5 --linear-size 8 --dimension 1 \
  --hopping 1.0 --interaction 2.0 --burn-in 500 --samples 3000 \
  --density-momenta '1' --density-frequency-max 8 \
  --density-measurements-per-block 30 \
  --density-continuation-dir density-continuation-v1 --no-trace
```

For multiple dimensions, separate components with commas and momentum rows
with semicolons, for example `--density-momenta '1,0;0,1'`. The measurement
count must provide at least two complete blocks. Existing bundle directories
are never overwritten.

To export exact selected imaginary-time correlations instead, replace
`--density-frequency-max` with comma-separated canonical lags. This selects
the mutually exclusive `imaginary_time_lag` basis without introducing a
retained time grid:

```sh
./build/dev/examples/qmc_interacting_demo \
  --particles 6 --beta 1.5 --linear-size 8 --dimension 1 \
  --hopping 1.0 --interaction 2.0 --burn-in 500 --samples 3000 \
  --density-momenta '1' --density-lags 0,0.25,0.5,0.75 \
  --density-measurements-per-block 30 \
  --density-continuation-dir density-lag-continuation-v1 --no-trace
```
