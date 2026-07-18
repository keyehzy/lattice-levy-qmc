# C++ ideal-gas port

This directory contains the C++20 port of the ideal, non-interacting sampler.
It implements conditioned free lattice bridges, finite-torus kernels and
traces, canonical partition recursion, labeled permutation cycles, explicit
winding sectors, and dense ideal-world-line configuration assembly.

The build requires CMake 3.24 or newer, a C++20 compiler, Ninja, and GSL 2.7
or newer. Tests use an installed GoogleTest 1.17 when available and otherwise
download its pinned upstream revision.

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

Release builds do not download test dependencies:

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
```

`worldlines` stores torus coordinates, while `worldlines_covering` retains
unwrapped coordinates and therefore the winding information. `64` is the
number of retained time links per interval `beta`; it is an observation
resolution, not a Trotter discretization.

The example executable accepts `[N beta M L d t seed]`:

```sh
./build/dev/examples/qmc_ideal_demo 8 2.0 64 16 2 1.0 2026
```
