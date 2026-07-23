# Changelog

All notable changes to the C++ implementation are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add stable log-weight categorical sampling to `Random`, including explicit
  zero-mass and non-finite-input validation.
- Add `TorusLayout` and the strong `SiteId` physical-site identity for checked
  lattice volume, encoding, displacement, shifts, and neighborhood traversal.
- Add an immutable `CanonicalEnsemble` that binds a validated free model to its
  canonical recursion and supports explicit particle-number prefix reuse.
- Add an immutable `OneParticleSpectrum`, owned by `CanonicalEnsemble`, with a
  checked torus layout and reusable one-dimensional momentum, sine, and cosine
  tables.
- Add immutable `FreePathKernels`, owned by `CanonicalEnsemble`, to bind
  hopping, numerical work limits, and optional torus provenance for retained
  and continuous exact free-path operations.
- Add a validated `Permutation` value with authoritative successor storage and
  a deterministic read-only cycle decomposition.
- Add an owning `RetainedMeasurementContext` so equal-time and retained-density
  estimators can share one physical-site traversal and retained-grid provenance.
- Add typed equal-time, retained-density, retained-geometry, cycle-statistics,
  and winding accumulators that enforce grid and particle-count provenance and
  own sample normalization.
- Add a bundled interaction measurement that derives action, overlap, double
  occupancy, and energy estimators from one full overlap sweep.
- Add an owning `ContinuousMeasurementContext` with physical seam positions,
  stable globally ordered hops, and explicit equal-time event groups.
- Add validated `MatsubaraModeSet` and shape-safe `MatsubaraModeField` values
  for selected signed frequencies and torus momenta shared by retained and
  continuous measurements.
- Add `ContinuousMatsubaraPlan` with bounded binary64 phase reduction, exact
  seam/period behavior, stable interval transforms, and overflow-safe torus
  site phases.
- Add `ContinuousParticleModes` and exact grouped-event projection of
  unnormalised density residence amplitudes and signed bond-midpoint hopping
  flux, including canonical-density and winding-preserving zero modes.
- Add `DensityMatsubaraAccumulator` and the provenance-owning
  `ContinuousMatsubaraDensityCorrelations` result for analytically centred
  continuous-time density susceptibility and sampled mean amplitudes.
- Add `HoppingResponseAccumulator` and `HoppingResponse` for analytically
  centred signed-flux gauge response, axis-resolved diamagnetic terms, derived
  paramagnetic current correlations, and sampled mean-flux diagnostics.
- Add `ContinuousPairDensityModes` and exact occupancy-replay projection of
  unnormalised on-site pair-density Matsubara amplitudes, with atomic
  equal-time groups and an exact pair-overlap zero mode.
- Add self-describing segment, stitch, fixed-seam stitch-sweep, and random-seam
  stitch option values.
- Add generated compile-time version metadata in `<qmc/version.hpp>`, sourced
  from the CMake project version.
- Document the C++ refactoring audit, including correctness, performance,
  testing, packaging, and API follow-up work.

### Changed

- Make `Model` valid by construction with private physical parameters,
  read-only accessors, and a cached checked lattice volume; invalid scalar,
  particle-label, and volume inputs now fail at construction.
- Centralize bounded-tail growth, work limits, included-mass tracking, and tail
  acceptance for winding and conditioned Bessel-count laws; support expansion
  now preserves evaluated weights, and Bessel counts use an adjacent-term
  recurrence after one seeded weight.
- Route canonical-cycle, conditioned Bessel-count, and stitch-matching draws
  through the shared log-weight sampler without temporary probability buffers.
- Make retained density-correlation results valid by construction and bind
  their inverse temperature, time grid, and torus layout directly to the data;
  Matsubara transforms no longer accept a separately supplied model.
- Replace the retained Matsubara result's public parallel arrays with an owning
  `MatsubaraDensityCorrelations` class. Callers now use `grid()`, `modes()`,
  `values()`, and checked `at()` accessors; construction enforces the complete
  retained frequency and flat-momentum sequence.
- Make `ContinuousPath` a valid-by-construction value with private storage,
  read-only endpoint/event views, and structural equality; malformed paths are
  now rejected at construction instead of remaining mutable public records.
- Make `ContinuousConfiguration` a valid-by-construction value that owns its
  model provenance, topology, and private world-line storage; geometry queries
  now consume that provenance and interaction estimators reject a mismatched
  interacting model without revalidating every path.
- Store continuous-configuration topology as one `Permutation`; cycle views are
  now derived by that owner and stitch acceptance publishes topology with one
  non-throwing move instead of synchronizing two public vectors.
- Make retained ideal configurations valid by construction with one
  authoritative `Permutation` and one private covering-space world-line
  buffer; physical sites, cycle paths, and winding are now derived from those
  read-only values.
- Use the shared torus layout for retained observables, full and incremental
  interaction occupancy, stitch locality buckets, and example flat indexing.
- Route full pair-overlap evaluation through the continuous measurement
  event sweep, sharing time normalization, stable tie ordering, physical hop
  geometry, and equal-time grouping without changing interaction estimators.
- Route reusable ideal sampling and exact observables through
  `CanonicalEnsemble`; the ideal demo and interacting sampler now construct the
  canonical recursion once instead of rebuilding or accepting an independently
  supplied table.
- Reuse ensemble-owned spectrum data in canonical traces, thermodynamic
  derivatives, momentum energies, one-body density matrices, twisted
  partitions, and twist curvature without eagerly materializing all `L^d`
  momentum modes.
- Route retained bridges, continuous bridges, winding draws, interval updates,
  and stitch weights through ensemble-owned free-path kernels; scalar-heavy
  free functions remain available as one-off compatibility wrappers.
- Exclude GoogleTest translation units from clang-tidy analysis while retaining
  the existing project check profile for library and example targets.
- Prepare and validate complete sweep, run, and random-seam plans before their
  first update or random draw; repeated runs reuse one prepared stitch mixture.
- Route pair and collective stitch requests through one `stitch_update()` API
  selected by `StitchUpdateOptions::strand_count`; seeded stitch sequences may
  change, but the proposal and acceptance laws are unchanged.
- Route two-strand and collective stitch matchings through the same validated
  permanent recursion and row-major weight evaluation, removing the legacy
  pair-only draw while preserving its exact proposal law.
- Cache prepared torus-bridge winding distributions by physical displacement
  within each fixed-seam stitch sweep, sharing exact matching normalizations
  and covering-endpoint draws without changing the proposal law or seeded
  stream.
- Store automatic stitch selections in a fixed eight-label buffer and scan
  candidate spans directly, removing the particle-count-sized selection bitmap,
  heap-backed result, per-strand filtered candidate allocations, and the
  proposal-validation bitmap while preserving the seeded proposal stream.

### Removed

- Remove the mutable public `FreeBosonTable` and APIs that accepted a `Model`
  and canonical table separately, preventing mismatched physical provenance.
- Remove retained samples' duplicate torus/cycle path storage and copied
  `log_ZN`; canonical normalization remains owned by `CanonicalEnsemble`.
- Remove continuous samples' copied `log_Z0_N`; canonical normalization remains
  owned by `CanonicalEnsemble` rather than each sampled configuration.
- Remove the positional segment/stitch update overloads and the separate
  `k_stitch_update()` entry point in favor of named option values.

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
