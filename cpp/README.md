# C++ ideal-gas port

This directory contains the C++20 port of the ideal, non-interacting sampler.
It implements conditioned free lattice bridges, finite-torus kernels and
traces, canonical partition recursion, labeled permutation cycles, explicit
winding sectors, dense ideal-world-line configuration assembly, and exact or
retained-grid estimators for canonical thermodynamics, one-body, topology,
density, and imaginary-time observables.

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

Run the distribution-level midpoint, winding, cycle, and torus tests with:

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

qmc::Model model{
    .particle_count = 8,
    .beta = 2.0,
    .linear_size = 16,
    .dimension = 2,
    .hopping = 1.0,
};
qmc::Random random(2026);
const auto configuration =
    qmc::sample_ideal_boson_configuration(model, 64, random);
configuration.validate();
const auto table = qmc::canonical_table(model);
const auto thermodynamics = qmc::canonical_thermodynamics(model, table);
const auto momentum = qmc::momentum_distribution(model, table);
const auto equal_time = qmc::equal_time_observables(configuration);
const auto correlations = qmc::retained_density_correlations(configuration);
```

`worldlines` stores torus coordinates, while `worldlines_covering` retains
unwrapped coordinates and therefore the winding information. `64` is the
number of retained time links per interval `beta`; it is an observation
resolution, not a Trotter discretization.

See [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md) for estimator definitions,
normalizations, exactness, and retained-grid conventions.

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
