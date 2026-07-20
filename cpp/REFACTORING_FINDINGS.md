# C++ refactoring findings

Date: 2026-07-19

## Scope and baseline

This review covers the C++20 library, private implementation, tests, examples,
and CMake support under `cpp/`. It is a refactoring review, not a request to
preserve every current API or random stream. The non-negotiable constraint is
that the mathematical distributions, path conventions, topology, winding, and
observable results remain correct.

No production or test source was changed during this review. The only repository
artifact added is this report. Existing untracked `.DS_Store`, observable output,
and trace files were left untouched.

The current baseline is strong:

- the development build is current;
- all 71 fast tests pass;
- all 79 tests in the statistical preset pass;
- all 79 tests in the AddressSanitizer/UndefinedBehaviorSanitizer preset pass;
- the existing GCC 15 build is current and all 79 tests pass there;
- `format-check` passes;
- the implementation has unusually good invariant and distribution-level test
  coverage for numerical Monte Carlo code.

The main opportunity is therefore not cosmetic cleanup. It is to make the
invariants that the tests prove become properties of the types and ownership
model, then remove repeated validation, duplicated representations, and repeated
numerical setup that are currently needed to defend mutable public structs.

## Executive recommendation

The three immediate correctness guards are complete: public flat-buffer extents
use checked derived sizes, local occupancy replacements stage affected timelines
before an allocation-free commit, and canonical recursions are owned by the
validated model that created them. Broader reorganization can now proceed
without those known memory-safety, cache-desynchronization, or provenance
boundaries.

The best target architecture is three small, concrete ownership units:

1. `CanonicalEnsemble`: owns one validated `Model`, lattice/spectrum data, and
   its matching canonical recursion table. It supplies ideal samples and exact
   observables without accepting a separately supplied, possibly mismatched
   table.
2. `Configuration`/`ContinuousPath` value types: construct only valid states,
   expose read-only data and cheap views, and reserve full validation for import,
   tests, and debug audits.
3. `AcceptedChainState`: owns the continuous configuration, occupancy index,
   overlap, and action as one transactionally updated unit.

Those units naturally support the two most useful internal utilities:

- `TorusLayout`, which centralizes checked volume, modulo reduction, flat site
  indexing, displacement indexing, and neighborhood traversal;
- `PathCursor`/`PathSlice`, which traverses a sorted event path once and is
  shared by splitting, interval replacement, stitching, rotation, and
  measurements.

This is deliberately not a generic framework or class hierarchy. Each proposed
type collects one invariant that is currently spread across several files.

## Priority map

| Priority | Finding | Primary payoff | Estimated size |
| --- | --- | --- | --- |
| P0 | Make occupancy replacement transactional (done 2026-07-19) | Correctness and simpler move code | Medium |
| P0/P1 | Checked flat extents and `TorusLayout`/`SiteId` (done 2026-07-19); add grid provenance | Memory safety, shared geometry, and fewer allocations | Medium |
| P0 | Bind model and canonical table (done 2026-07-19); add reusable free numerics | Correctness and large repeated-work reduction | Medium |
| P0 | Make paths/configurations valid-by-construction (`ContinuousPath` and retained ideal configuration done 2026-07-20) | Ownership clarity and removal of nested validation | Large |
| P1 | Path cursor/slice migration (done 2026-07-20) | Simpler boundary semantics and faster path surgery | Medium |
| P1 | Add a retained-measurement context and accumulators | Less repeated work and a smaller demo | Medium |
| P1 | Collect reusable free-particle numerics | Numerical clarity and less repeated setup | Medium |
| P1 | Unify discrete log-weight and bounded-tail sampling | One truncation policy and fewer allocations | Medium |
| P1 | Unify full/incremental action around accepted state | One source of truth and faster full evaluation | Medium |
| P1 | Make permutation topology authoritative (continuous and retained ideal done 2026-07-20) | Remove duplicated state and synchronization code | Medium |
| P1 | Separate stitch selection, proposal, and commit | Readability, testability, and API clarity | Medium |
| P2 | Add streaming runs, checkpoints, and prepared options | Operational quality of life | Small/medium |
| P2 | Consolidate local helpers, names, and value equality | Less duplication and review noise | Small |
| P2 | Split large translation units and demo output code | Navigation and reviewability | Small |
| P2 | Exclude GoogleTest from tidy (done 2026-07-19); finish packaging, CI, and benchmarks | Consumer and maintainer quality of life | Medium |
| P2 | Isolate the GSL/process-global error policy | Safer library embedding | Small/medium |
| P2 | Add invariant builders and fault-boundary tests | Safer migration of owning abstractions | Small |

## Detailed findings

### 1. P0: make local action replacement a real transaction

Status (2026-07-19): complete. `ReplacementTransaction` copies only affected
site timelines into a staged overlay, calculates the proposed overlap there,
and publishes prepared values/nodes only through a non-throwing commit. Normal
and stitch replacements now share one proposal-finalization path using paired
`PathReplacementView` values. Action, topology, and accepted-counter failures,
local rejection, transaction abandonment, and mid-replacement failure have
dedicated ledger/state/cache regressions.

Pre-refactor evidence:

- `OccupancyIndex::replace_paths()` mutates timelines incrementally while
  removing and adding paths (`src/occupancy_index.cpp:172-192`).
- Both callers invoke it before entering their `try` blocks
  (`src/interacting_sampler.cpp:604` and `src/interacting_sampler.cpp:771`).
- Rollback calls the same validating, checked, map-mutating `adjust_path()`
  operations in reverse (`src/occupancy_index.cpp:114-145` and
  `src/occupancy_index.cpp:195-202`), so it can itself allocate or throw while
  handling another failure.
- The current catch/rollback logic only protects action calculation and the
  Metropolis draw after `replace_paths()` returns
  (`src/interacting_sampler.cpp:605-617` and `src/interacting_sampler.cpp:772-783`).
- Accepted-counter capacity is checked after the ledger has entered proposal
  state and outside that catch/rollback region
  (`src/interacting_sampler.cpp:619` and
  `src/interacting_sampler.cpp:786-790`). A counter-overflow exception can
  therefore create the same state/cache split even when `replace_paths()` itself
  completed.
- The ledger API takes separate adjacent `old_paths` and `new_paths` spans and
  does not express their one-to-one relationship in its type
  (`src/occupancy_index.hpp:20-25`). Callers build and keep two parallel pointer
  vectors in sync manually.
- The accept/reject transaction is duplicated between ordinary replacements
  and stitch replacements.

If allocation, checked arithmetic, or timeline rebuilding throws partway through
`replace_paths()`, the accepted configuration remains unchanged but its occupancy
ledger may already be partially changed. This is an exception-safety hole and is
also why the move implementations contain a large amount of manual ordering.
The original porting guide explicitly calls out the invariant that allocation or
action failure must not desynchronize accepted state and caches
(`../docs/CPP_PORT.md:359-370`).

Recommendation:

- Have `OccupancyIndex::begin_replacement(old, new)` return a small RAII
  `ReplacementTransaction` containing the proposed overlap and a staged overlay
  of affected timelines. Prefer doing all allocation and arithmetic on that
  overlay, leaving the accepted ledger untouched until `commit()` swaps prepared
  nodes into place.
- If an in-place undo log is retained instead, precompute every key/event entry
  and make rollback allocation-free and non-throwing; a destructor cannot safely
  depend on copying `Site` keys, inserting map entries, or revalidating paths
  while another exception is active.
- Keep counter-capacity checks, action calculation, and topology preparation
  inside the transaction lifetime. Commit only after all potentially throwing
  preparation is complete.
- Route both normal and stitch moves through one `try_proposal()` implementation;
  the proposal supplies replacements and an optional topology commit.
- Pass a span of `PathReplacementView{label, old_path, new_path}` to the
  transaction instead of parallel spans. This gives size/pairing one source of
  truth and makes accidentally reversed arguments impossible.
- After sorting replacements, detect invalid/duplicate labels with an adjacent
  comparison rather than allocate an `N`-element `vector<bool>` for every local
  move (`src/interacting_sampler.cpp:583-600`).

Verification:

- retain the current global rejection-preserves-state test, and add forced local
  segment/stitch rejection cases that compare the occupancy ledger as well as
  the public configuration and scalar caches;
- add fault-injection tests that throw during removal, addition, action
  calculation, and topology preparation, then compare the ledger with the full
  event-sweep evaluator;
- test counter semantics for invalid, failed, rejected, and accepted proposals.

### 2. P0: bind a canonical table to the model that created it and reuse it

Status (2026-07-19): complete. `CanonicalEnsemble` now owns one validated
`Model` and its private, fully checked canonical recursion. Sampling and exact
observables consume the ensemble rather than independently supplied model/table
pairs; model-only convenience overloads construct a temporary ensemble. The
ideal demo and interacting sampler retain one ensemble for repeated sampling.
Read-only recursion views and explicit bounded prefix queries preserve the valid
smaller-particle-number use case without losing physical provenance. Seeded
retained-grid and continuous-time tests prove reusable and one-off sampling are
identical. In four warm release-build trials, 200 samples at `N=256`, `L=4`,
`d=1`, and `M=1` took 0.0545-0.0554 s through a retained ensemble versus
0.0755-0.0763 s through the one-off wrapper. This focused result confirms the
expected setup-reuse win; it is not an end-to-end production benchmark.
Spectrum and trigonometric workspace caching remains under finding 7.

Pre-refactor evidence:

- `FreeBosonTable` contains only two public vectors (`include/qmc/free_boson.hpp:15-20`).
  Consumers can check capacity and finiteness but cannot prove that `beta`, `L`,
  dimension, or hopping match, or even that both vectors came from one
  consistent recursion. A longer table is legitimately usable for a smaller
  particle-number query, so size is capacity rather than provenance.
- Table validation is separately implemented in
  `src/continuous_configuration.cpp:31-41` and `src/observables.cpp:17-35`.
- `sample_ideal_boson_configuration()` always calls `canonical_table(model)`
  (`src/configuration.cpp:241`) and has no table-reuse overload.
- The ideal demo already computes a canonical table once
  (`examples/ideal_demo.cpp:799`) but `sample_ensemble()` ignores it and rebuilds
  the table for every sample (`examples/ideal_demo.cpp:248-250`).
- The documented library-use sequence also samples first and then constructs a
  second table for exact observables because the public retained sampler offers
  no reuse path (`README.md:79-84`).
- The continuous sampler already demonstrates the intended reuse overload
  (`src/continuous_configuration.cpp:345-357`).

Recommendation:

Introduce an immutable `CanonicalEnsemble` (or `FreeBosonWorkspace`) built from a
validated `Model`. It should own the model, one-particle spectrum/trigonometric
workspace, `log_z`, and `log_Z`, and expose:

- ideal retained-grid and continuous samples;
- cycle sampling;
- thermodynamics, momentum, one-body, cycle, and twist observables;
- read-only spans for callers that truly need the tables.

Construction should check the complete recursion for finite/allowed values once
and fail at the numerical source of an overflow. Downstream methods should not
revalidate trusted vectors or discover a malformed table through `inf - inf` in
a probability calculation.

Keep free-function convenience wrappers if useful, but implement them by creating
a temporary ensemble. Repeated workflows and both samplers should retain one
ensemble.

A focused release-build microbenchmark with `N=256`, `L=4`, `d=1`, `M=1`
measured six invocations of 20 canonical-table constructions at 0.0020-0.0032 s
and 20 complete ideal samples (which currently include that construction) at
0.0074-0.0116 s. This is intentionally not a production performance claim, but
it shows that redundant setup is a material fraction of a small retained sample.

The porting guide already lists table and momentum-cosine caching as its first
likely performance win (`../docs/CPP_PORT.md:414-425`); the current measurements
are enough to move it from a hypothetical to an actionable refactor.

Verification:

- add a regression that a table/workspace cannot be paired with a different
  set of physical free-particle parameters; explicitly test the chosen policy
  for reusing a table prefix at smaller particle number;
- compare seeded samples from the old convenience wrapper and reusable object;
- benchmark independent-sample throughput for small `M` and moderate `N`.

### 3. P0: replace mutable record bags with valid-by-construction values

Status (2026-07-20): the `ContinuousPath` and retained ideal-configuration
slices are complete. `ContinuousPath` construction now
owns and validates duration, endpoint dimensions, sorted event times, axes,
directions, coordinate range, and endpoint consistency. Path storage is private;
callers receive read-only endpoint references and an event span, and structural
equality is defined for paths and events. Trusted queries, cursor operations,
interaction sweeps, and occupancy transactions no longer revalidate complete
path storage. `DenseWorldlines` now owns immutable shape metadata and private flat
storage, and `IdealBosonConfiguration` validates its private model, retained-grid
size, topology, covering geometry, endpoint joining, and representable winding
at construction. Its `validate()` remains an explicit diagnostic audit. Torus
positions and cycle geometry are derived from the read-only covering buffer.
Construction-failure tests and the existing boundary, surgery, sampling,
topology, observable, action, and cache tests cover the migrations. `Model` and
the remaining `ContinuousConfiguration` storage remain open.

Pre-refactor evidence:

- `Model` is documented as immutable but all fields are public and mutable
  (`include/qmc/model.hpp:21-32`).
- `DenseWorldlines`, `IdealBosonConfiguration`, `ContinuousPath`, and
  `ContinuousConfiguration` expose their shape, topology, endpoints, event, and
  cache fields publicly (`include/qmc/configuration.hpp:14-48`,
  `include/qmc/path.hpp:17-35`, and
  `include/qmc/continuous_configuration.hpp:15-26`).
- `DenseWorldlines::at()` consequently validates the entire declared shape on
  every element access (`src/configuration.cpp:180-189`).
- `ContinuousPath::position_at()` validates the complete path before scanning it
  (`src/path.cpp:238-249`).
- configuration observables each repeat full validation
  (`src/observables.cpp:584-585`, `src/observables.cpp:642-643`,
  `src/observables.cpp:724-726`, and `src/observables.cpp:768-770`).
- `pair_overlap_time()` validates the configuration, then its detail function
  validates the model and every path again (`src/interaction.cpp:199-206` and
  `src/interaction.cpp:154-168`).

Recommendation:

- Keep simple input parameter structs if designated initialization is valued,
  but convert them once into validated internal values (`ValidatedModel`,
  `TorusLayout`, prepared numerical options).
- Give `ContinuousPath` constructors/factories that validate and normalize event
  ordering, endpoint consistency, duration, and dimension once; expose events
  through `std::span<const JumpEvent>`.
- Encapsulate dense shape fields and compute offsets without revalidating the
  buffer on every access.
- Make configuration topology and paths private/read-only. Use explicit builders
  for sampling/import and controlled replacement methods for the chain.
- Preserve an expensive `validate()` as a diagnostic/debug audit, not a required
  subroutine of every trusted internal query.
- Standardize failure categories at the same boundary: malformed imported or
  public input should be `invalid_argument` (or one validation exception),
  internal invariant failure should be `logic_error`/an assertion, and numerical
  work/representation limits should retain their descriptive runtime/overflow
  errors. The current mutable records blur these categories.

The goal is not to remove validation. It is to validate at trust boundaries and
make invalid intermediate states unrepresentable inside the library.

### 4. P0/P1: check flat extents, then centralize geometry in `TorusLayout`

Status (2026-07-19): the checked-flat-extent and geometry slices are complete.
Density, Matsubara, and retained-geometry buffers derive their sizes with
checked products. `TorusLayout` now owns checked volume/strides, strict and
covering-space encoding, decoding, flat displacement, coordinate shifts, and
periodic neighborhood traversal. Full action, transactional occupancy,
retained measurements, stitch buckets, and demo indexing use its strong
`SiteId` keys instead of independently flattened integers or vector-valued
physical sites. Round-trip, displacement, shift, neighborhood uniqueness,
overflow, and equal-volume/different-layout tests cover the shared geometry;
the existing action/cache and stitch tests cover its migrated consumers. A
shape-aware lattice field and retained-grid provenance remain open.

Evidence:

- flatten/decode logic is private to observables
  (`src/observables.cpp:37-90`), another site flattener exists in interaction
  (`src/interaction.cpp:24-42`), and demo code repeats stride construction
  (`examples/ideal_demo.cpp:135-145`).
- torus distance and recursive neighborhood enumeration are embedded in the
  sampler orchestration file (`src/interacting_sampler.cpp:36-215`).
- full overlap uses an `unordered_map<size_t, ...>` keyed by flat sites
  (`src/interaction.cpp:62-68`), while the incremental index uses
  `std::map<Site, ...>` (`src/occupancy_index.hpp:50-53`) and stitch buckets use
  an `unordered_map<Site, ...>` with a third site representation
  (`src/interacting_sampler.cpp:68-80`).
- `displacement_index()` allocates origin, target, and displacement vectors per
  call (`src/observables.cpp:80-90`) and is called inside particle-pair and
  time-lag hot loops (`src/observables.cpp:226-232` and
  `src/observables.cpp:654-661`).
- flattened time/lattice result sizes use unchecked `time_points * volume`
  products in density, Matsubara, and geometry code
  (`src/observables.cpp:648-652`, `src/observables.cpp:681-705`, and
  `src/observables.cpp:730-735`). This is also used in a public input-shape
  check, where unsigned wrap could make malformed storage look valid.
- `ImaginaryTimeDensityCorrelations` carries only a spatial point count. The
  Matsubara transform checks that count against `model.volume()`, so different
  lattices with the same volume (for example `L=4,d=1` and `L=2,d=2`) are
  accepted even though their displacement geometry and Fourier modes differ.
  A different `beta` with the same shape is also accepted and changes the
  frequencies and integration weight.

The public Matsubara input check is an immediate correctness issue, not merely a
cleanup opportunity. With `volume == 2`, for example, setting `time_points` to
`size_t_max / 2 + 1` and supplying an empty vector makes the product wrap to
zero, so the current check passes; the following nested loops then index that
empty vector. Replace all such extent arithmetic with checked products before
undertaking the larger layout migration.

Recommendation:

Add one small `TorusLayout` value containing `L`, dimension, checked strides, and
volume. It should provide reduction, `encode`, `decode_into`, flat displacement,
and neighborhood iteration. Define a strong `SiteId` wrapper for flat physical
sites, while retaining `Site` for covering-space coordinates.

Use `SiteId` consistently in the full action workspace, occupancy index, stitch
buckets, retained positions, and lattice fields. This removes vector-key
allocation/comparison from the occupancy ledger and removes a surprising amount
of coordinate boilerplate. All layout-derived allocation sizes should go
through this layout's checked shape helpers, and other derived extents should
use the existing checked-math utilities. Validate externally supplied flat
buffers with the same checked extent before indexing.

Avoid a general tensor framework. A `LatticeField<T>` with a layout and checked
flat storage is enough to make public flattened observable results easier to use.
Time-dependent fields should additionally carry one immutable retained-grid
descriptor (`beta` and point count), so transforms consume their input's
provenance rather than an unrelated `Model` argument.

Verification should include encode/decode/displacement round trips, maximum-size
boundary tests, a crafted wrapped Matsubara shape that is rejected before
allocation or indexing, and rejection of a same-volume/different-layout field.

### 5. P1: traverse continuous paths once with `PathCursor` and `PathSlice`

Status (2026-07-20): complete. A private monotone `PathCursor` records the
positions and event ranges immediately before and through each cut. `PathSlice`
can select those endpoint sides explicitly, retaining the default
right-continuous `(tau0, tau1]` convention while also expressing the
`[shift, beta]` and `[0, shift)` pieces required by time-origin rotation.
Splitting, interval resampling, stitch splicing, and rotation now traverse each
source path once and assemble results through shared slice materialization,
replacement, splicing, and concatenation utilities. Stitch proposals reuse the
same cuts for endpoint weights and path assembly. Rotation precomputes both
slices for every worldline before joining permutation successors. Exact
equivalence regressions cover the previous stitch and rotation traversals,
coincident events at both seams, covering-space suffix translation, preserved
RNG streams for interval resampling, zero-duration splitting, and the permitted
few-ulp worldline-duration drift from `beta`.

Pre-refactor evidence:

- `position_at()` validates and scans from the beginning for every query
  (`src/path.cpp:238-249`).
- `split_continuous_path()` scans every event for every output piece and calls
  `position_at()` twice per piece (`src/path.cpp:416-457`).
- `resample_path_interval()` validates once, calls `position_at()` twice (each
  validating again), then scans the old events twice
  (`src/path.cpp:460-498`).
- stitch proposal computes cut positions (`src/interacting_sampler.cpp:708-717`),
  then `splice_path_interval()` recomputes them and rescans the event ranges
  (`src/interacting_sampler.cpp:433-486`).
- time-origin rotation contains another hand-written event slicing operation
  (`src/continuous_configuration.cpp:205-266`).

Recommendation:

Introduce a cursor over already validated, sorted events. It should yield the
left/right position and iterator range at requested monotone cuts. Build a
`PathSlice` from those iterators and endpoint positions, then share a single
splice/builder implementation between split, resample, stitch, and rotate.

This also gives the right-continuous boundary convention one implementation
instead of several similar combinations of `lower_bound`, `upper_bound`, `<=`,
and `>`.

In five back-to-back runs of a focused benchmark, splitting a 2,060-event path
at 99 cuts 20 times took 0.0238-0.0278 s. The absolute time is small, but the
current work is proportional to pieces times events; a cursor makes it
proportional to events plus pieces and is easier to reason about.

### 6. P1: share retained measurement state and provide real accumulators

Evidence:

- `equal_time_observables()` and `retained_density_correlations()` independently
  call `retained_positions()` (`src/observables.cpp:584-590` and
  `src/observables.cpp:641-648`).
- the ideal demo then calls equal-time, density-correlation, and geometry
  estimators separately for every sample (`examples/ideal_demo.cpp:302-315`),
  triggering repeated configuration validation and coordinate traversal.
- equal-time measurement materializes both all retained positions and an
  `M x volume` occupancy matrix even though occupancies are consumed one slice
  at a time (`src/observables.cpp:92-104` and `src/observables.cpp:584-614`).
- `SampleAverages` has 27 manually initialized/added/divided fields
  (`examples/ideal_demo.cpp:193-243` and `examples/ideal_demo.cpp:318-342`).
  Adding a result field requires updating several distant blocks correctly.
- flattened time/space arrays expose dimensions next to raw vectors rather than
  through shape-aware accessors (`include/qmc/observables.hpp:100-131`).
- the Matsubara conversion is a pair of direct DFTs and repeatedly allocates
  decoded coordinate vectors and complex phases inside its nested loops
  (`src/observables.cpp:686-719`).

Recommendation:

- Construct one `RetainedMeasurementContext` per configuration. It validates
  once and holds/reuses flat retained positions, a slice occupancy buffer, and
  optional precomputed displacement/phase data.
- Let equal-time, density, and geometry estimators consume that context.
- Add typed accumulator objects with `observe(context)` and `finish(count)`;
  keep raw per-configuration estimators for users who need them.
- Stream slices where possible instead of retaining `M x volume` intermediate
  occupancy state.
- Give flattened lattice/time results shape-aware accessors. For small
  particle-number/cycle tables, prefer a vector of row records (or an `at(n)`
  row view) over several public parallel vectors whose sizes can diverge.

The density correlation itself is quadratic in time origins and particle pairs
in the current direct estimator, while its subsequent direct time transform is
quadratic in retained time points and the spatial transform is quadratic in
lattice volume. Avoiding allocations in `displacement_index()` and sharing a
separable `LatticeTransformPlan` are the first changes; an FFT dependency or
FFT-based occupancy autocorrelation should be selected only after workload
benchmarks.

Release-demo measurements for 100 samples at `beta=2`, `t=1`, `N=8`, `L=16`,
`d=2` scaled from 0.56 s at `M=32`, to 1.62 s at `M=64`, to 5.22 s at `M=128`.
At `M=64`, changing `N=4,8,16` measured 0.63, 1.58, and 4.81 s. These end-to-end
results confirm the expected roughly quadratic retained-measurement/transform
scaling and make this a useful benchmark target; they do not isolate
correlation from DFT cost.

### 7. P1: collect free-particle numerics into reusable kernels

Evidence:

- `canonical_table()` allocates a new `terms` vector for every particle number
  and revalidates the model through `log_one_particle_trace(duration, model)` for
  every cycle length (`src/free_boson.cpp:123-151`).
- `sample_cycle_labels()` allocates log/probability buffers on every cycle, then
  copies the remaining labels, allocates an `N`-bit selection mask, and erases by
  scanning the full active set (`src/free_boson.cpp:167-206`). The many
  one-cycle case is consequently quadratic in label-management work in addition
  to the required canonical probability work.
- canonical log recursion is implemented again for twisted boundaries
  (`src/observables.cpp:474-516`).
- the derivative recursion in thermodynamics
  (`src/observables.cpp:262-297`) is substantially repeated in twist curvature
  (`src/observables.cpp:530-581`).
- momentum angles, cosines, and exponential weights are rebuilt in the trace,
  beta derivatives, mode energies, one-body matrix, twist partition, and twist
  curvature.
- `occupation_moments()` allocates log-term vectors for every momentum mode
  (`src/observables.cpp:176-203`).
- public midpoint/bridge functions repeatedly accept adjacent durations,
  hopping, lattice size, step count, and numerical options as raw positional
  values (`include/qmc/free_numerics.hpp:23-55` and
  `include/qmc/path.hpp:38-65`). Several are mutually convertible, and every
  call restates parameters that are constant for a model.

Recommendation:

- Add a `OneParticleSpectrum` owned by `CanonicalEnsemble`, with momentum
  components/energies and reusable one-dimensional trigonometric tables.
- Add a validated `FreePathKernels` context (owned by the ensemble/sampler) that
  binds hopping, numerical limits, and optional torus layout while continuing
  to take `Random&` explicitly. Use small `BridgeRequest`/`Interval` values at
  the longest public call sites instead of strings of convertible scalars; keep
  concise free-function wrappers for one-off use.
- Implement one canonical recursion taking a span or callback of log cycle
  weights; reuse a scratch buffer or streaming `log_add_exp` rather than allocate
  one vector per row.
- Represent first/second log derivatives with a tiny `Jet2`/`LogDerivatives`
  value and run them through one recurrence. A dedicated two-derivative value is
  preferable to a general automatic-differentiation framework.
- Compute isotropic per-axis twist curvature once; the demo currently repeats
  the full calculation for every axis (`examples/ideal_demo.cpp:801-807`).
- Maintain one active-label array during cycle construction: sample selected
  labels into a prefix with swaps and remove that prefix without a full-size
  mask/copy. Keep the anchor and directed-cycle law explicit in tests because a
  changed label ordering may intentionally change the seeded stream.

This refactor should precede low-level optimization because it establishes the
cache boundaries and removes repeated trigonometry naturally.

### 8. P1: unify discrete log-weight and bounded-tail sampling

Evidence:

- cycle sampling converts log probabilities to a new probability vector before
  calling `Random::discrete_index()` (`src/free_boson.cpp:173-186`).
- permanent matching repeats the same conversion row by row
  (`src/stitch_matching.cpp:82-100`), while pair stitching has a third custom
  log-space draw (`src/interacting_sampler.cpp:410-430`).
- the permanent builder returns a raw vector, so its immediately following
  sampler revalidates the matrix and every table entry that the library just
  produced (`src/stitch_matching.cpp:40-80`).
- `log_add_exp` is private to stitch matching even though `log_sum_exp` is a
  public numerical primitive (`src/stitch_matching.cpp:27-35`).
- symmetric winding and displaced torus winding independently implement initial
  support, `1.5x/+8` growth, Bessel evaluation, geometric tail bounds, limits,
  and signed support construction (`src/free_boson.cpp:12-89` and
  `src/path.cpp:54-195`). Bessel pair counts contain a third adaptive support
  loop (`src/free_numerics.cpp:157-221`).

Recommendation:

- Add `Random::discrete_log_index(span<double>)` (or a free function taking the
  RNG) with centralized validation and normalization.
- Centralize `log_add_exp` and allow canonical/permanent recursions to accumulate
  without temporary probability vectors.
- Extract a narrow `AdaptiveDiscreteSupport` helper that owns support growth,
  work limits, included mass, and tail-bound acceptance. Keep the mathematical
  weight/tail formulas as named functions for each distribution.
- Preserve already evaluated weights when support grows. All three current
  loops rebuild the full prefix after each expansion.
- For conditioned Bessel counts, append log weights with the adjacent-term
  ratio after one seeded value instead of evaluating two `lgamma` calls for
  every old and new index on every growth. Keep the recurrence formula beside
  the tail bound and cover it with the existing exact-PMF tests.
- Use fixed-capacity arrays for permanent state, matching columns, and row
  probabilities: `k <= 8`, so the maximum table has only 256 entries. This
  removes several tiny heap allocations per stitch without obscuring the
  recursion (`src/stitch_matching.cpp:40-104`).
- Wrap those arrays and `k` in a private `PreparedPermanent` value so shape and
  finiteness are checked once at the external-input boundary, not again between
  two adjacent trusted functions.

Do not hide the truncation derivations behind templates. The useful abstraction
is the shared control loop and failure policy; the distribution-specific bound
should remain visible and independently tested.

### 9. P1: use the same site/event abstractions in full and incremental action

Evidence:

- full overlap copies every event into `TimedEvent` and stable-sorts all events
  even though every path's events are already sorted
  (`src/interaction.cpp:104-123`).
- the full evaluator and incremental ledger use different site keys and separate
  occupancy-update logic (`src/interaction.cpp:62-148` versus
  `src/occupancy_index.cpp:114-163`).
- sampler construction first computes full overlap, then independently rebuilds
  the occupancy index (`src/interacting_sampler.cpp:529-538`).
- an accepted global proposal computes full overlap and then rebuilds the index
  (`src/interacting_sampler.cpp:952-969`).
- the public action, interaction-energy, total-energy, and double-occupancy
  helpers independently trigger full overlap/validation work when a caller asks
  for several quantities (`src/interaction.cpp:199-252`).
- rejected new paths can cause `timeline(position)` to create zero-valued site
  timelines; rollback removes counts but does not prune empty map entries.
- the sampler stores both `pair_overlap_` and the exactly derived
  `action_ = U * pair_overlap_`, while the ledger is a third representation of
  the same accepted-state quantity. Local moves update the scalars by repeated
  floating-point deltas; rebuilt ledgers do not rebase them.
- `interaction_detail.hpp` still describes its arbitrary-path evaluator as a
  proposal-evaluation seam, but no sampler path calls it; the sampler includes
  that header without using it. This is stale design surface from before the
  incremental ledger.

Recommendation:

- Use `SiteId` from `TorusLayout` and one occupancy transition helper in both
  paths.
- Replace the global sort with a small min-heap/k-way merge over the already
  sorted per-path event spans. This reduces auxiliary event copying and changes
  sorting from `O(E log E)` to `O(E log N)`. Give the heap an explicit
  `(time, path index, event index)` tie-break so equal-time transitions retain
  the existing stable path/event order and remain reproducible.
- Let an `AcceptedChainState` build its occupancy index once and derive
  `pair_overlap`/`action` from that same index. Keep the independent full sweep as
  an audit/reference implementation.
- Cache at most the overlap (with its ledger generation) and derive action from
  immutable `U` on access. Use compensated delta accumulation or a configurable
  periodic rebase from the ledger for long chains, with the full evaluator as an
  independent drift check.
- Add a pure `measure_interaction(configuration, model)` bundle (or an overload
  accepting a previously computed overlap summary) so non-sampler callers can
  obtain all interaction observables with one full sweep.
- Prune empty timelines or use a flat hash map keyed by `SiteId`.
- Either keep the arbitrary-path full evaluator private as an explicitly named
  test oracle or exercise it as a real production fallback; remove the unused
  private header/include and outdated proposal comment in either case.

The independent evaluator remains valuable for tests; sharing site reduction and
event transition semantics does not require sharing the whole algorithm.
This keeps the porting guide's intended split: merge sorted streams in the
production evaluator while retaining the global sort as a reference oracle
(`../docs/CPP_PORT.md:414-431`).

### 10. P1: make permutation topology authoritative

Status (2026-07-20): the continuous and retained ideal slices are complete. A validated
`Permutation` now owns authoritative successors and a private deterministic
cycle cache, exposing both only through read-only views. `ContinuousConfiguration`
stores one private topology value instead of parallel public cycle and successor
vectors. Sampling, time rotation, cycle updates, and stitch construction consume
that value. Stitch preparation validates the complete proposed permutation while
the occupancy replacement is staged, and acceptance publishes topology with one
non-throwing move. Construction, deterministic decomposition, invalid-successor,
endpoint-connectivity, accepted/rejected stitch, and transaction-failure tests
cover the migration. `IdealBosonConfiguration` now uses the same authoritative
topology and stores only one private covering-space retained-worldline buffer.
The sampler no longer retains per-cycle labels/base/winding/path records, a
dense torus copy, or `log_ZN`; physical sites, cycle winding, and cycle geometry
are derived without changing the sampled stream. Construction, topology,
endpoint, winding, empty-system, seeded reusable/one-off equivalence, observable,
and statistical tests cover the retained migration.

Evidence:

- `ContinuousConfiguration` stores both `cycles` and `permutation`
  (`include/qmc/continuous_configuration.hpp:16-20`), then validation proves they
  agree (`src/continuous_configuration.cpp:43-76`).
- accepted stitches explicitly assign both representations
  (`src/interacting_sampler.cpp:794-795`).
- ideal configurations additionally store per-cycle labels, permutation, cycle
  covering/torus paths, and two dense worldline buffers
  (`include/qmc/configuration.hpp:30-45`). Validation spends substantial code
  proving all of those copies agree (`src/configuration.cpp:52-161`).
- After assembly, no production estimator reads `IdealCyclePath::base_point` or
  `torus_path`; they are retained only so validation can compare them with data
  already present in the covering/dense representations.
- `log_ZN` and `log_Z0_N` are model-wide constants copied into every
  configuration. Outside construction, validation, rotation, and tests, no
  production consumer reads either field.

Recommendation:

- Add a validated `Permutation` value with `successor(label)`, deterministic
  cycle iteration, and optional lazily cached cycle decomposition.
- Store only that authoritative topology in continuous configuration. Cycle
  moves can request a cycle view; stitch commits only a successor update.
- For retained ideal configurations, select one authoritative geometry. A good
  default is covering-space retained worldlines plus permutation/cycle metadata;
  reduce to torus coordinates in a measurement context or an optional lazy
  cache. Avoid permanently storing cycle paths, dense paths, and torus copies
  unless profiling proves each is independently needed.
- Remove per-configuration partition constants once `CanonicalEnsemble` owns
  them. If callers need the normalization, expose it from the ensemble/sample
  provenance rather than duplicating it in every mutable state.

This is a data-model change, so migrate one representation at a time and retain
the existing exhaustive validation tests as equivalence tests during the move.

### 11. P1: split stitch mechanics and replace positional APIs with option values

Evidence:

- `interacting_sampler.cpp` is 1,036 lines. Its first 490 lines combine hashing,
  torus neighborhoods, partner selection, mixture preparation, interval
  selection, kernel evaluation, matching, and path splicing.
- public stitch functions take five to seven adjacent optional/numeric
  parameters (`include/qmc/interacting_sampler.hpp:97-129`). Call sites contain
  sequences such as `nullopt, 0.25, 1, 0.05` whose meaning is not locally clear.
- fraction/probability/count validation is repeated between single stitch,
  k-stitch, and stitch sweep (`src/interacting_sampler.cpp:217-241`,
  `src/interacting_sampler.cpp:272-311`, `src/interacting_sampler.cpp:362-382`,
  `src/interacting_sampler.cpp:806-875`, and
  `src/interacting_sampler.cpp:877-908`).
- a fixed-seam sweep correctly reuses its left positions and buckets
  (`src/interacting_sampler.cpp:910-923`), but proposal construction recalculates
  selected left positions and path cuts.
- `stitch_log_weights()` evaluates a complete torus bridge distribution for
  every matrix entry (`src/interacting_sampler.cpp:385-405`); after matching,
  `sample_continuous_bridge_torus()` evaluates the selected distributions again
  (`src/interacting_sampler.cpp:728-737`).
- strand selection repeatedly allocates `vector<bool>` and filtered candidate
  vectors even though `k <= 8` (`src/interacting_sampler.cpp:244-311`).
- the weight evaluation and matching draw both special-case `k == 2` solely to
  retain a legacy evaluation/RNG order (`src/interacting_sampler.cpp:385-430`).
  That duplicates the general permanent path even though seeded stream identity
  is not a mathematical requirement.

Recommendation:

- Move pure topology/geometry selection into `stitch_selection.cpp` and bridge
  matching/splicing into `stitch_proposal.cpp`; keep sampler orchestration and
  commit in `interacting_sampler.cpp`.
- Replace long signatures with `SegmentUpdateOptions` and `StitchUpdateOptions`.
  Prepare/validate `StitchMixture` once when a sweep plan is installed.
- Add `StitchSeamContext` containing `tau0/tau1`, left positions, and partner
  buckets. Its invariance across accepted fixed-seam stitches is part of the
  type's documented contract.
- Let the seam context cache a small `TorusBridgeDistribution` by endpoint
  displacement. The same object should expose its log normalization and sample
  a covering endpoint, so matrix construction and selected bridge generation do
  not repeat Bessel support work.
- For at most eight strands, use a small fixed array of selected labels and scan
  candidate spans instead of allocating an `N`-bit vector and copied filtered
  vectors on each attempt.
- Make the two-strand convenience wrapper construct k-stitch options and use the
  same implementation.
- Route `k == 2` through the same prepared-permanent evaluator and sampler as all
  other supported sizes. Preserve the pair-law test, but allow its seeded sample
  sequence to change.

A release-demo diagnostic using `U=0`, `beta=1`, `t=1`, `N=16`, `L=4`, `d=2`
and 5,000 random-seam macro-steps with one stitch attempt per step took
0.22-0.23 s for `k=2`, 0.35-0.36 s for `k=4`, and 0.82-0.87 s for `k=8` across
three runs. The macro-step also includes two time rotations and trace output, so
this does not isolate permanent or bridge cost; it is a compact regression
workload for the proposed seam and bridge-distribution caches.

### 12. P2: improve sampling workflow APIs

Evidence:

- `run()` always materializes every `InteractingObservables` record in a vector
  (`src/interacting_sampler.cpp:1013-1033`). The interacting demo reimplements
  burn-in, thinning, output, and averaging so it can stream a trace
  (`examples/interacting_demo.cpp:165-205`).
- the recommended random-seam `A B^m A` kernel is not expressible as one
  `SweepOptions`; the demo therefore holds stitch scheduling fields separately
  from its `SweepOptions` and runs both manually
  (`examples/interacting_demo.cpp:20-32` and
  `examples/interacting_demo.cpp:165-173`).
- there is no statistics reset/iteration helper, checkpoint format, or explicit
  statement of sampler copy semantics. Copying a sampler duplicates the RNG and
  rebuilds the occupancy index (`src/interacting_sampler.cpp:541-558`).
- move metadata is duplicated in `move_index()` and `move_name()`, and the
  statistics array has a hard-coded length of five
  (`src/interacting_sampler.cpp:20-34`, `src/interacting_sampler.cpp:492-506`, and
  `include/qmc/interacting_sampler.hpp:141-170`).
- options are validated only along the branch that happens to execute. Empty
  systems return success before checking explicit invalid segment/stitch
  arguments (`src/interacting_sampler.cpp:629-650` and
  `src/interacting_sampler.cpp:843-891`), while a numerical failure is counted
  as a stitch attempt but not as a segment attempt.
  These behaviors are currently accidental API semantics rather than one stated
  policy.
- `sweep()` performs segment and cycle updates before a later stitch call
  validates its options (`src/interacting_sampler.cpp:974-991`), and
  `random_seam_stitch_sweep()` rotates once before validating the fixed-seam
  request (`src/interacting_sampler.cpp:942-949`). Invalid plans can therefore
  throw after partially advancing the chain.
- the interacting trace header records column names but not the model, seed,
  sweep plan, burn-in, or thinning that produced it
  (`examples/interacting_demo.cpp:178-201`), making a detached output file hard
  to reproduce.

Recommendation:

- Add `run(sample_count, options, observer)` or a sample generator/range so
  callers can stream, average, or persist without materializing all records.
  This is also the callback/output-iterator shape originally recommended in the
  porting guide (`../docs/CPP_PORT.md:296-304`).
- Define a concrete `SweepPlan` that can include segment, cycle, global, time
  shift, and random-seam macro-kernels in documented order. Avoid a generic
  command framework.
- Validate and prepare that plan once, before any RNG consumption, and define
  whether invalid/no-op moves and proposal-generation failures count as
  attempts. Apply that policy uniformly across system sizes and move kinds.
- Add `reset_statistics()` and iteration over `{MoveKind, name, statistics}`.
- Prefer a move-only sampler unless deterministic RNG-fork copy behavior is a
  deliberate documented feature. Provide an explicit `clone()` if it is.
- Design a versioned checkpoint containing configuration, RNG state, cached
  values (or enough to rebuild them), and run counters for long QMC jobs.
- Give the example trace writer a small metadata preamble with model, seed,
  prepared plan, and schema version; check final stream state so a truncated
  result is reported rather than presented as a completed run.
- Use a `MoveKind::Count` sentinel or one constexpr metadata table to derive
  indexing, names, and storage size.

### 13. P2: simplify names, validation helpers, and local implementation

Concrete low-risk cleanup:

- `configuration.cpp` locally reimplements `checked_product`, `checked_scale`,
  and `checked_add` despite `src/checked_math.hpp`
  (`src/configuration.cpp:12-44`).
- `free_numerics.cpp` reimplements `coordinate_offset`,
  `coordinate_from_offset`, and `displacement`, which already exist in
  `checked_math.hpp` (`src/free_numerics.cpp:29-49`).
- finite/nonnegative duration and hopping checks are repeated in free numerics,
  free boson, path, model, and interacting code. Prepared parameter values would
  remove most; a small shared validation helper can cover the remainder.
- names for the same concept vary: `log_ZN`, `log_Z0_N`, `pair_overlap`,
  `pair_overlap_time`, `time_links`, and `time_links_per_beta`.
- tests hand-implement field-by-field equality for paths, configurations, and
  retained samples (`tests/test_interacting_sampler.cpp:25-50` and
  `tests/test_configuration.cpp:11-29`) because value records do not define
  structural equality.
- `interaction_action()` accepts a chemical potential even though the public
  comment says it is constant in this fixed-N ensemble
  (`include/qmc/interaction.hpp:13-16`). Remove it from the core action or expose
  the constant offset as a separately named calculation.
- prefer half-open size loops and checked `count + 1` construction. Broad
  clang-tidy currently reports many unsigned `<=` loops as potentially wrapping,
  even where surrounding checks make them safe.
- default structural `operator==` for validated model, event, path, and topology
  values where exact identity is meaningful. Keep approximate numerical-result
  comparisons as explicitly named test matchers rather than weakening equality.

Apply these as mechanical follow-ups after the owning abstractions land; doing
them first would create churn in code that is likely to move.

### 14. P2: split measurement implementation and examples by responsibility

Evidence:

- `observables.cpp` is 807 lines and mixes exact canonical thermodynamics,
  spectra, one-body response, topology statistics, retained estimators, and
  discrete Fourier transforms.
- `ideal_demo.cpp` is 858 lines and combines CLI parsing, ensemble accumulation,
  table serialization, gnuplot program generation, process launching, and
  console summaries.
- output code reconstructs lattice coordinates from momentum records because
  lattice layout is not a shared public utility
  (`examples/ideal_demo.cpp:487-516` and `examples/ideal_demo.cpp:530-585`).
- the demo defaults write `ideal_observables/` and `interacting_trace.dat` into
  the caller's working directory, while `cpp/.gitignore` ignores only the build
  tree and user presets. Following the source-directory examples leaves dozens
  of generated artifacts in `git status`.

Recommendation:

- split implementation into `canonical_observables.cpp`,
  `retained_observables.cpp`, and `lattice_transforms.cpp` while keeping the
  public header stable initially;
- split the ideal tool into a thin `main`, `IdealEnsembleAccumulator`, table
  writers, and plot-script writer;
- share only genuinely common example helpers (model CLI options, checked output
  file, exit handling) in an example-support target. Do not move demo-specific
  plotting or file formats into the core library.
- Point repository examples at an ignored build-tree output directory and ignore
  the exact documented default artifact paths (or require an explicit output
  path). Keep broad `*.dat`/`*.png` patterns out of the project-wide ignore so
  intentional reference data can still be committed.

### 15. P2: finish the library build/consumer story

Status (2026-07-19): the GoogleTest-noise slice is complete. Test executables
still compile in the tidy preset but no longer opt into clang-tidy, preventing
GoogleTest macro expansions from dominating its output. Library and example
targets retain the existing `.clang-tidy` checks, header filter, and warning
policy unchanged. Installation, dependency-fetch policy, CI, and benchmark
targets remain open.

Evidence:

- targets advertise an `$<INSTALL_INTERFACE:include>` but there are no
  `install()`, export, or package-config rules (`CMakeLists.txt:33-38` and
  `CMakeLists.txt:56-63`). A consumer cannot actually install and
  `find_package()` the library.
- examples and tests may fetch dependencies during configure
  (`examples/CMakeLists.txt:1-12`, `tests/CMakeLists.txt:1-17`), with no explicit
  offline/fetch policy switch.
- there is no repository CI configuration, so Clang/GCC, sanitizer, format, and
  statistical presets are local conventions rather than enforced gates.
- The project intentionally enables broad
  bugprone/modernize/performance/readability checks with `WarningsAsErrors: ''`
  (`.clang-tidy:2-16`). Before this slice, the same profile was attached to all
  GoogleTest targets, whose macro expansions produced the bulk of the output
  and obscured findings in project code.

Recommendation:

- add install/export rules, generated version/config files, and a tiny external
  consumer test using `find_package(qmc CONFIG REQUIRED)`;
- add `QMC_FETCH_DEPENDENCIES` (default suitable for developers, disabled for
  packaging/offline builds) and fail with a clear package instruction when it is
  off;
- add CI for Clang and GCC fast tests, format, the project clang-tidy profile,
  and a scheduled/statistical sanitizer job;
- retain the project clang-tidy profile for libraries and examples while
  excluding macro-heavy GoogleTest targets; add a separate test profile only if
  direct test-source analysis becomes useful;
- add an optional benchmark target for canonical setup/sampling, path slicing,
  retained correlations, stitch proposals, and full/local action evaluation.

### 16. P2: isolate GSL and its process-global error policy

Evidence:

- the only direct GSL use is the scaled modified Bessel wrapper in
  `src/free_numerics.cpp:85-110`;
- before evaluating it, the library permanently disables GSL's process-global
  error handler through `std::call_once` (`src/free_numerics.cpp:93-99`);
- this is necessary to observe underflow through the `_e` return code, but it
  also changes error behavior for unrelated GSL use in the host process. It is
  not robust isolation in the other direction either: if a host replaces the
  global handler after the one-time call, later QMC evaluations no longer
  enforce the policy they rely on.

Recommendation:

- keep all GSL access behind one private numerical backend rather than exposing
  GSL through additional files;
- document the current process-global side effect immediately if GSL remains;
- evaluate a side-effect-free scaled-Bessel implementation/backend. Swapping the
  global handler around individual calls is not a thread-safe fix when a host
  application may call GSL concurrently;
- add direct backend tests for underflow, large order, convolution, and the
  parameter ranges exercised by both winding samplers.

This is not a reason to replace a proven numerical routine casually. It is a
library integration boundary that should be explicit and narrow.

### 17. P2: refactor tests around invariant builders and fault boundaries

The existing tests should be preserved; they are the safety net that makes the
larger data-model changes reasonable. Add a small test-support library rather
than broad fixtures:

- builders for valid `Model`, `ContinuousPath`, and small configurations;
- reusable `expect_valid_permutation`, `expect_same_path`, and
  `expect_cache_matches_full_action` helpers;
- parameterized boundary-semantics cases for events at `0`, cuts, and `beta`;
- ownership, read-only recursion, and bounded prefix tests for `CanonicalEnsemble`;
- wrapped-product and maximum-representable shape tests for every public flat
  result, plus same-volume/different-layout and different-grid provenance tests;
- exception/fault-injection tests for replacement transactions;
- explicit rejected segment, cycle, and stitch tests that prove configuration,
  ledger, overlap, action, topology, and counters follow the chosen semantics;
- a direct random-seam macro-kernel test that checks the two rotations, fixed-seam
  attempt count, topology/cache invariants, and the documented palindromic
  schedule rather than relying only on the demo;
- either deterministic copy/RNG-fork tests for `InteractingSampler` or removal of
  implicit copying in favor of an explicitly named clone;
- equivalence tests between cursor-based and existing path surgery during
  migration;
- benchmark baselines separate from pass/fail correctness tests.

Also give tests a configured private include path rather than reaching into
`../src/stitch_matching.hpp` (`tests/test_interacting_sampler.cpp:1`). This keeps
private-kernel tests without coupling them to the current directory layout.

Avoid hiding mathematical expected values behind helpers. The current tests are
valuable partly because their Python/exact-enumeration reference values remain
visible at the assertion sites.

## Suggested migration sequence

1. Completed 2026-07-19: checked public flat extents, local rejection and
   injected-failure regressions, `ReplacementTransaction`, and unified
   normal/stitch proposal finalization using the existing representation.
2. Completed 2026-07-19: added reusable `CanonicalEnsemble`, bound canonical
   provenance, and routed the ideal demo and both samplers through it while
   retaining model-only convenience wrappers.
3. `TorusLayout`/`SiteId` and checked-flat-shape portions completed 2026-07-19;
   shared validation-helper cleanup and lattice-field/grid provenance remain.
4. Completed 2026-07-20: `PathCursor`/`PathSlice` migration for split,
   resample, stitch, and time-origin rotation, with exact traversal and boundary
   equivalence tests.
5. Completed 2026-07-20: `ContinuousPath` and retained dense/configuration
   encapsulation, plus authoritative continuous and retained `Permutation`
   topology; full validation remains available as an explicit diagnostic audit.
6. Add retained measurement context and accumulators, then optimize direct
   correlation kernels from benchmark evidence.
7. Consolidate canonical derivative/log-distribution numerics.
8. Split translation units and demos after responsibilities have stabilized.
9. GoogleTest exclusion from clang-tidy completed 2026-07-19; add install, CI,
   benchmark support, and an external consumer test.

Each step should keep the current deterministic, invariant, and statistical
tests green. Refactors that intentionally change the seeded stream should compare
distribution-level results and exact invariants rather than individual samples.

## What not to do

- Do not merge retained-grid ideal paths with sparse continuous paths; their
  representations and meaning of time resolution are intentionally different.
- Do not introduce a sampler class hierarchy. Ideal independent sampling and
  finite-interaction Markov updates share numerical services, not lifecycle.
- Do not turn every numerical kernel into a template. Keep probability laws and
  tail bounds readable in ordinary functions.
- Do not cache data independently in several public structs. Cache inside the
  owner whose invariant proves when it is valid.
- Do not optimize away the independent full action evaluator or exhaustive
  validation tests; use them as audit oracles for the optimized path.

## Review evidence and benchmark notes

Commands used against the unchanged source tree:

```sh
cmake --build --preset dev -j 4
ctest --preset dev --output-on-failure
ctest --preset statistical --output-on-failure
ctest --preset sanitizers --output-on-failure
cmake --build --preset dev --target format-check
cmake --build --preset tidy -j 4
cmake --build build/gcc -j 4
ctest --test-dir build/gcc --output-on-failure
```

The timing observations above used the existing release libraries/executable and
temporary binaries/output under `/tmp`; they did not alter repository source.
They are directional measurements on one machine, not committed performance
budgets. Before implementing a performance-motivated change, add a reproducible
benchmark workload and compare results plus statistical/invariant tests.
