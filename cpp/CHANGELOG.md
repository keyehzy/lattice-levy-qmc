# Changelog

All notable changes to the C++ implementation are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `TorusLayout` and the strong `SiteId` physical-site identity for checked
  lattice volume, encoding, displacement, shifts, and neighborhood traversal.
- Add an immutable `CanonicalEnsemble` that binds a validated free model to its
  canonical recursion and supports explicit particle-number prefix reuse.
- Add a validated `Permutation` value with authoritative successor storage and
  a deterministic read-only cycle decomposition.
- Add generated compile-time version metadata in `<qmc/version.hpp>`, sourced
  from the CMake project version.
- Document the C++ refactoring audit, including correctness, performance,
  testing, packaging, and API follow-up work.

### Changed

- Make `ContinuousPath` a valid-by-construction value with private storage,
  read-only endpoint/event views, and structural equality; malformed paths are
  now rejected at construction instead of remaining mutable public records.
- Store continuous-configuration topology as one `Permutation`; cycle views are
  now derived by that owner and stitch acceptance publishes topology with one
  non-throwing move instead of synchronizing two public vectors.
- Make retained ideal configurations valid by construction with one
  authoritative `Permutation` and one private covering-space world-line
  buffer; physical sites, cycle paths, and winding are now derived from those
  read-only values.
- Use the shared torus layout for retained observables, full and incremental
  interaction occupancy, stitch locality buckets, and example flat indexing.
- Route reusable ideal sampling and exact observables through
  `CanonicalEnsemble`; the ideal demo and interacting sampler now construct the
  canonical recursion once instead of rebuilding or accepting an independently
  supplied table.
- Exclude GoogleTest translation units from clang-tidy analysis while retaining
  the existing project check profile for library and example targets.

### Removed

- Remove the mutable public `FreeBosonTable` and APIs that accepted a `Model`
  and canonical table separately, preventing mismatched physical provenance.
- Remove retained samples' duplicate torus/cycle path storage and copied
  `log_ZN`; canonical normalization remains owned by `CanonicalEnsemble`.

### Fixed

- Make local interacting-path replacements strongly transactional by staging
  affected occupancy timelines and committing state, topology, action caches,
  and counters only after all potentially throwing preparation succeeds.

## [0.5.0] - 2026-07-19

### Added

- Generalize random-seam topology updates from pairs to configurable
  `k`-strand proposals for `k <= 8`.
- Sample endpoint matchings by their free weights using permanent-based dynamic
  programming.
- Add configurable strand-count mixtures and topology-change statistics to the
  interacting driver.

## [0.4.0] - 2026-07-19

### Added

- Add reversible random-seam stitch updates that reconnect closed world-line
  strands without opening a path.
- Add a dynamic occupancy index for local interaction-action evaluation.
- Add stitch scheduling and locality controls to the interacting driver.
- Document the random-seam heat-bath derivation and detailed-balance argument.

## [0.3.0] - 2026-07-19

### Added

- Add the `qmc::interacting` library for continuous-time finite-interaction
  Bose-Hubbard sampling.
- Add sparse jump-event paths, continuous configurations, interaction-action
  evaluation, and segment, cycle, and global Metropolis updates.
- Add an interacting trace driver with energy, double-occupancy, winding, and
  acceptance summaries.
- Add deterministic invariants and an optional exact-diagonalization
  statistical regression for the interacting sampler.

## [0.2.0] - 2026-07-19

### Added

- Add canonical thermodynamics, momentum occupations, cycle and winding
  statistics, equal-time measurements, retained-time correlations, and
  Matsubara transforms for ideal configurations.
- Expand the ideal example into a sampling and plotting driver with
  machine-readable output tables.
- Document estimator definitions, normalization, and retained-grid semantics.

## [0.1.0] - 2026-07-18

### Added

- Add the initial C++20 `qmc::ideal` library with model validation, free
  numerical kernels, canonical cycle sampling, bridge construction, winding
  sectors, and retained-grid configurations.
- Add explicit seeded random-number handling and public umbrella headers.
- Add a cxxopts-based ideal-gas command-line example, defaulting to a
  two-dimensional model.
- Add CMake presets, strict warning configuration, formatting, clang-tidy,
  sanitizer support, and deterministic and statistical GoogleTest suites.

[Unreleased]: https://github.com/keyehzy/lattice-levy-qmc/compare/fb7de70bec89899afd646808abd6350c67fab409...HEAD
[0.5.0]: https://github.com/keyehzy/lattice-levy-qmc/compare/d790b9b6ebd577b536a52946112d8581ce61cae6...fb7de70bec89899afd646808abd6350c67fab409
[0.4.0]: https://github.com/keyehzy/lattice-levy-qmc/compare/6b92d28ae260f1e22273b053a06d5ed1e4b5e831...d790b9b6ebd577b536a52946112d8581ce61cae6
[0.3.0]: https://github.com/keyehzy/lattice-levy-qmc/compare/2ce98e8369d0b73ba784b92418e1ab838da02c9c...6b92d28ae260f1e22273b053a06d5ed1e4b5e831
[0.2.0]: https://github.com/keyehzy/lattice-levy-qmc/compare/6ae5036ec4c4e00856ad3e2a4193268d71795947...2ce98e8369d0b73ba784b92418e1ab838da02c9c
[0.1.0]: https://github.com/keyehzy/lattice-levy-qmc/compare/4031ea7a718910cb9c2c0e9fb8476f0c3da6eb65...6ae5036ec4c4e00856ad3e2a4193268d71795947
