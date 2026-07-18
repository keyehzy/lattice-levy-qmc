# C++ ideal-gas port

This directory contains the C++20 port of the ideal, non-interacting sampler.
The scaffold currently provides build and test infrastructure; production QMC
code has not been added yet.

Configure, build, and test from this directory:

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

The first test-enabled configuration uses an installed GoogleTest 1.17 or
downloads the pinned upstream revision. Release builds do not download test
dependencies:

```sh
cmake --preset release
cmake --build --preset release
```
