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
| P0/P1 | Checked flat extents and `TorusLayout`/`SiteId` (done 2026-07-19); retained-grid provenance (done 2026-07-20) | Memory safety, shared geometry, and fewer allocations | Medium |
| P0 | Bind model and canonical table (done 2026-07-19); reusable path kernels (done 2026-07-21) | Correctness and large repeated-work reduction | Medium |
| P0 | Make paths/configurations valid-by-construction (done 2026-07-21) | Ownership clarity and removal of nested validation | Large |
| P1 | Path cursor/slice migration (done 2026-07-20) | Simpler boundary semantics and faster path surgery | Medium |
| P1 | Add a retained-measurement context (done 2026-07-20) and equal-time/density accumulators (done 2026-07-21); add remaining accumulators | Less repeated work and a smaller demo | Medium |
| P1 | Collect reusable free-particle numerics (spectrum, path-kernel, partition-recurrence, cycle-sampling, and occupation-moment slices done 2026-07-21) | Numerical clarity and less repeated setup | Medium |
| P1 | Unify discrete log-weight draws and bounded-tail sampling (done 2026-07-20) | One truncation policy and fewer allocations | Medium |
| P1 | Unify full/incremental action around accepted state (ownership slice done 2026-07-20) | One source of truth and faster full evaluation | Medium |
| P1 | Make permutation topology authoritative (continuous and retained ideal done 2026-07-20) | Remove duplicated state and synchronization code | Medium |
| P1 | Separate stitch selection, proposal, and commit (option/prevalidation slice done 2026-07-21) | Readability, testability, and API clarity | Medium |
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

Status (2026-07-21): complete. The `ContinuousPath`, retained
ideal-configuration, continuous-configuration, and `Model` slices are complete.
`ContinuousPath` construction now
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
`ContinuousConfiguration` now likewise owns its validated model provenance,
authoritative topology, and private path vector; it has no default invalid state,
exposes only read-only views, and defines structural equality. Construction
checks shape, duration, and endpoint connectivity once, while `validate()`
retains the complete path audit. Geometry queries and time rotation consume the
owned model, interaction estimators reject mismatched interacting-model
provenance, and `AcceptedChainState` is the only path/topology mutation boundary.
The unused per-sample `log_Z0_N` copy is removed in favor of the normalization
owned by `CanonicalEnsemble`. `Model` now converts a designated
`ModelParameters` input record into private, read-only physical parameters,
rejecting invalid scalars, particle-label overflow, and unrepresentable lattice
volume at construction. Its checked volume is cached and downstream trusted
code no longer revalidates the model. Construction-failure tests and the
existing boundary, surgery, sampling, topology, observable, action, and cache
tests cover the migrations.

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

Status (2026-07-20): the checked-flat-extent, geometry, and retained-grid
provenance slices are complete.
Density, Matsubara, and retained-geometry buffers derive their sizes with
checked products. `TorusLayout` now owns checked volume/strides, strict and
covering-space encoding, decoding, flat displacement, coordinate shifts, and
periodic neighborhood traversal. Full action, transactional occupancy,
retained measurements, stitch buckets, and demo indexing use its strong
`SiteId` keys instead of independently flattened integers or vector-valued
physical sites. Round-trip, displacement, shift, neighborhood uniqueness,
overflow, and equal-volume/different-layout tests cover the shared geometry;
the existing action/cache and stitch tests cover its migrated consumers. A
valid-by-construction `ImaginaryTimeDensityCorrelations` now owns its checked
flat storage together with an immutable `RetainedGrid` containing beta, retained
point count, and `TorusLayout`. The Matsubara transform consumes that provenance
directly instead of accepting an unrelated `Model`; exact equal-volume 1D/2D
transform and stored-beta regressions cover the mismatch boundary. A general
shape-aware lattice-field abstraction remains a possible measurement follow-up.

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

Status (2026-07-21): the retained-measurement-context and equal-time/density
accumulator slices are complete.
`RetainedMeasurementContext` owns the retained-grid provenance and flattened,
time-major physical positions derived once from a retained configuration.
Equal-time and retained-density estimators consume it, while their existing
configuration overloads remain as one-off wrappers. The ideal demo constructs
one context per sample, and equal-time occupancy measurement now reuses one
volume-sized slice buffer instead of retaining an `M x volume` matrix.
`EqualTimeAccumulator` and `RetainedDensityCorrelationAccumulator` bind one
retained grid and particle count, reject mismatched contexts before mutation,
own their sample counts, and return normalized results from `finish()`. The
ideal demo uses both instead of manually initializing, adding, and dividing
their result fields. Exact context/wrapper equivalence, two-sample averages,
single-sample identity, grid/particle mismatches, bounds, and empty boundaries
cover the new ownership units. Cycle/winding/geometry accumulators, shape-aware
result accessors, and transform plans remain open.

Pre-context evidence:

- `equal_time_observables()` and `retained_density_correlations()` independently
  call `retained_positions()` (`src/observables.cpp:584-590` and
  `src/observables.cpp:641-648`).
- the ideal demo then calls equal-time, density-correlation, and geometry
  estimators separately for every sample (`examples/ideal_demo.cpp:302-315`),
  triggering repeated configuration validation and coordinate traversal.
- equal-time measurement materializes both all retained positions and an
  `M x volume` occupancy matrix even though occupancies are consumed one slice
  at a time (`src/observables.cpp:92-104` and `src/observables.cpp:584-614`).

Remaining evidence:

- `SampleAverages` still manually initializes, adds, and normalizes its cycle,
  winding, and retained-geometry fields (`examples/ideal_demo.cpp:188-247` and
  `examples/ideal_demo.cpp:301-347`). Adding one of those result fields requires
  updating several distant blocks correctly.
- flattened time/space arrays expose dimensions next to raw vectors rather than
  through shape-aware accessors (`include/qmc/observables.hpp:134-143` and
  `include/qmc/observables.hpp:228-248`).
- the Matsubara conversion is a pair of direct DFTs and repeatedly allocates
  decoded coordinate vectors and complex phases inside its nested loops
  (`src/observables.cpp:769-815`).

Recommendation:

- Completed 2026-07-20: construct one `RetainedMeasurementContext` per
  configuration. It validates once and owns the flattened retained positions.
  Optional precomputed displacement/phase data remains a transform-plan
  follow-up.
- Let equal-time, density, and geometry estimators consume that context.
- Completed 2026-07-21 for equal-time and retained-density results: add typed
  accumulator objects with `observe(context)` and sample-count-owning
  `finish()`; keep raw per-configuration estimators for users who need them.
  Cycle, winding, and retained-geometry accumulation remain separate follow-ups.
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

Status (2026-07-21): the canonical derivative-recurrence slice is complete. One
private recurrence now propagates first and second log derivatives for both
canonical thermodynamics and twist curvature, with one reusable probability
scratch buffer instead of one allocation per recursion row. Thermodynamics
retains its per-row nonnegative roundoff handling, while twist propagation
preserves the signed second derivative until converting it to free-energy
curvature. Exact Fock-space thermodynamics, twist finite differences, isotropic
axis equality, invalid-axis handling, and the empty-system boundary cover the
shared implementation. The spectrum/trigonometric slice is also complete. An
immutable `OneParticleSpectrum` owned by `CanonicalEnsemble` now holds the
checked torus layout and `O(L)` wavevector, sine, and cosine tables. Complete
`L^d` momentum records remain demand-driven, so sampling-only ensemble
construction does not acquire a volume-sized cache. Canonical traces reuse one
exponent scratch buffer, while thermodynamic derivatives, mode energies,
one-body kernels, twisted partitions, and twist curvature consume the shared
tables. Construction/shape/energy failures, exact spectrum values, direct
one-body Fourier and twisted-angle equivalence, the existing Fock-space and
twist-curvature regressions, and matching-seed ensemble reproducibility cover
the boundary. In three warm Apple Clang 17 release-build trials, 80
repetitions of thermodynamics, momentum distribution, one-body density,
twisted partition, and twist curvature at `N=64`, `L=64`, `d=1`, `beta=1.25`,
and `t=0.9` took 0.0709-0.0717 s before the cache and 0.0180-0.0193 s after it.
This focused result demonstrates the repeated-work reduction; it is not an
end-to-end sampling benchmark. The reusable free-path-kernel slice is also
complete. Immutable `FreePathKernels` now bind hopping, bounded-tail controls,
and optional torus layout provenance once; `CanonicalEnsemble` owns that value,
and retained, continuous, segment, cycle, stitch, and global sampling consume
it directly. One-off scalar-heavy APIs remain wrappers. Exact matching-seed
tests cover Bessel counts, midpoint and retained bridges, winding, covering and
torus continuous bridges, torus kernel weights, complete configurations, and
subsequent RNG position under both owned and wrapper paths. The base/twisted
partition-recurrence slice is also complete. One private span-based recurrence
now owns the `log Z_0 = 0` base case, validates finite-or-negative-infinity
cycle weights and finite row results, and reuses one maximum-row scratch buffer.
Both `CanonicalEnsemble` construction and twisted-boundary evaluation consume
it instead of maintaining separate recursions and allocating one terms vector
per row. Direct base-case, closed-form prefix, zero-weight, and invalid-result
tests cover the kernel; the Python reference, permutation enumeration, and
twisted direct-angle tests retain end-to-end coverage, with an explicit
zero-twist/untwisted equivalence regression. The cycle-sampling workspace slice
is also complete. `CanonicalEnsemble::sample_cycles()` now allocates one active
label array and one maximum-sized log-weight scratch buffer per complete sample.
Partial Fisher-Yates selection forms each directed cycle in place, and bounded
unordered compaction removes the selected labels without a per-cycle pool,
full-sized mask, or full surviving-label scan. Repeated partition checks and a
complete four-particle labeled-permutation distribution regression cover the
canonical minimum-label root, directed-cycle, and compaction laws. Active-label
ordering, and thus labeled results for a fixed seed, may change; the cycle-length
and complete labeled-permutation laws are unchanged. In five alternating warm
Apple Clang 17 release-build trials, 1,000 samples in a many-singleton workload
at `N=512`, `L=16384`, `d=1`, `beta=0.01`, and `t=1` took 0.3576-0.3905 s
before the change and 0.2851-0.2926 s after it. This focused workload exercises
the previous worst-case label-management path; it is not an end-to-end sampling
benchmark.
The occupation-moment workspace slice is also complete. One particle-count-sized
log-term buffer is now allocated per complete momentum distribution and reused
for both the occupation and factorial-moment reductions of every mode. The
calculation keeps the previous term order and log-sum-exp formulas. Exact
Fock-space and momentum-normalization tests retain their existing coverage, and
a single-particle regression covers the empty factorial-moment span. In ten
alternating warm Apple Clang 17 release-build trials, 5,000 momentum-distribution
evaluations at `N=64`, `L=64`, `d=1`, `beta=1.25`, and `t=0.9` reduced measured
allocations per evaluation from 257 to 130 and median runtime from 0.2165 s to
0.1979 s. This focused allocation-counting workload demonstrates the intended
workspace reuse; it is not an end-to-end observable benchmark.

Recommendation:

- Completed 2026-07-21: add a `OneParticleSpectrum` owned by
  `CanonicalEnsemble`, with demand-derived mode energies and reusable
  one-dimensional trigonometric tables.
- Context portion completed 2026-07-21: add a validated `FreePathKernels`
  context (owned by the ensemble/sampler) that binds hopping, numerical limits,
  and optional torus layout while continuing to take `Random&` explicitly.
  Concise free-function wrappers remain for one-off use. Small
  `BridgeRequest`/`Interval` values remain a possible follow-up if the longest
  public wrapper call sites prove error-prone.
- Completed 2026-07-21: implement one canonical recursion taking a span of log
  cycle weights and reuse one scratch buffer rather than allocate one vector per
  row. Ordinary and twisted partitions now share that private kernel.
- Represent first/second log derivatives with a tiny `Jet2`/`LogDerivatives`
  value and run them through one recurrence. A dedicated two-derivative value is
  preferable to a general automatic-differentiation framework.
- Compute isotropic per-axis twist curvature once; the demo currently repeats
  the full calculation for every axis (`examples/ideal_demo.cpp:801-807`).
- Completed 2026-07-21: maintain one active-label array during cycle
  construction, sample selected labels into a prefix with swaps, and remove
  that prefix without a full-size mask/copy. Canonical minimum-label rooting and
  the complete directed labeled-permutation law are explicit in tests; changed
  active-label ordering intentionally permits a changed labeled seeded stream.
- Completed 2026-07-21: allocate one maximum-sized occupation-moment log-term
  buffer per momentum distribution and reuse its leading spans for every mode's
  occupation and factorial-moment reductions.

This refactor should precede low-level optimization because it establishes the
cache boundaries and removes repeated trigonometry naturally.

### 8. P1: unify discrete log-weight and bounded-tail sampling

Status (2026-07-20): the log-weight categorical-sampling slice is complete.
`Random::discrete_log_index()` now validates finite-or-negative-infinity inputs,
normalizes after subtracting the maximum, and rejects empty or zero-mass laws
before consuming randomness. Canonical cycle lengths, conditioned Bessel pair
counts, permanent-matching rows, and the two-strand stitch choice consume log
weights directly instead of allocating and normalizing probability buffers or
maintaining a separate log-space draw. Exact validation, extreme-offset,
deterministic-support, seeded offset-equivalence, and distribution tests cover
the shared boundary. Seeded streams may change where callers previously used a
different CDF convention or skipped deterministic draws; the probability laws
are unchanged. The bounded-tail slice is also complete. A private
`AdaptiveDiscreteSupport` now owns support growth, hard work limits, included
log mass, and the relative tail-tolerance decision while leaving each named
distribution's weights and geometric tail bound beside its sampler. Symmetric
winding, displaced torus winding, and conditioned Bessel-count samplers append
only newly exposed weights when support grows. Conditioned Bessel counts seed
one log weight and extend it with the adjacent-term ratio instead of repeatedly
evaluating `lgamma` over the full prefix. Controller, work-limit, finite-ring
kernel, exact conditioned-count distribution, midpoint, winding, bridge, and
statistical tests cover the shared policy and all three laws. Centralized
`log_add_exp` remains open. The fixed-capacity prepared-permanent slice is also
complete. A private `PreparedPermanent` owns its validated log-weight matrix and
maximum 256-state recursion table, then samples into a fixed-capacity matching
with fixed-capacity row scratch and without revalidating the matrix or table.
Exact recursion, input ownership, invalid-input, unique-matching, and
distribution tests cover the boundary. The legacy two-strand draw was routed
through this same prepared-permanent path in finding 11 on 2026-07-21.

Pre-slice evidence:

- cycle sampling converts log probabilities to a new probability vector before
  calling `Random::discrete_index()` (`src/free_boson.cpp:173-186`).
- permanent matching repeats the same conversion row by row
  (`src/stitch_matching.cpp:82-100`), while pair stitching has a third custom
  log-space draw (`src/interacting_sampler.cpp:410-430`).

Pre-prepared-permanent evidence:

- the permanent builder returns a raw vector, so its immediately following
  sampler revalidates the matrix and every table entry that the library just
  produced (`src/stitch_matching.cpp:40-80`).
- `log_add_exp` is private to stitch matching even though `log_sum_exp` is a
  public numerical primitive (`src/stitch_matching.cpp:27-35`).

Pre-bounded-tail evidence:

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

Status (2026-07-20): the accepted-state ownership slice is complete. A private
`AcceptedChainState` now owns the continuous configuration, occupancy index,
and cached pair overlap. Initial, global-proposal, and time-rotated states build
one occupancy ledger and take their overlap from that ledger instead of running
the independent full sweep and then rebuilding the same state separately. Local
paths, optional topology, affected timelines, and the proposed overlap are
published together through one non-throwing accepted-state transaction; an
abandoned transaction leaves the complete owner untouched. The action is now
derived from immutable `U` and the accepted overlap rather than retained as a
second scalar cache. If delta arithmetic crosses a few ulps below the physical
zero boundary, the transaction rebases from its fully staged occupancy ledger
before acceptance. Construction/global/time-shift ledger authority, derived
action, copied-owner, rejection, counter, topology, and action-failure tests
cover the boundary. The independent full event sweep remains the audit oracle.
The bundled public interaction-measurement slice is also complete as of
2026-07-21. `InteractionMeasurement` returns action, pair overlap, double
occupancy, kinetic/interaction/total energy, and event count from one full
overlap sweep. The sampler routes its accepted-state overlap cache through the
same scalar calculation without invoking that sweep. Exact piecewise,
model-provenance, empty/single-particle, and sampler-cache equivalence tests
cover the public and cached paths.

K-way event-merge experiment (2026-07-21): reverted. The implementation kept
one event per path in a deterministic heap instead of flattening and
stable-sorting every event, but the measured tradeoff did not justify the extra
machinery as the general production path:

- Sparse workload—256 paths, 8 events/path: k-way merge was about 5.5–7.5%
  faster and reduced auxiliary event storage from `O(E)` to `O(N)`.
- Dense workload—256 paths, 2,048 events/path: k-way merge was roughly 60%
  slower, despite better asymptotic complexity, because the flat stable sort
  has better locality and lower per-event overhead.

The flat stable sort therefore remains the production implementation. Revisit a
k-way or hybrid evaluator only if representative workload profiling shows that
event-buffer memory or sparse-path sorting is a material bottleneck and also
defines a measured crossover policy.

Numerical follow-up note (2026-07-20): the seeded finite-interaction invariant
test exposed cancellation at the physical zero boundary. At sweep 11, segment
update 2, the accepted overlap was `0.11278807006915575`; the incremental
`current - removed + added` evaluation produced
`-1.6653345369377348e-16`, while both the complete proposed occupancy-ledger
evaluation and the independent full event sweep returned exactly zero. The
current implementation responds to any finite negative delta result by
recomputing from the complete staged ledger and then applying the ordinary
finite/nonnegative validation. This is a defensive numerical rebase, not an
exact proof that every negative candidate is roundoff. A stricter follow-up
should prove zero structurally from integer occupancies on every
positive-duration interval, canonicalize to zero only at that boundary, and
treat a negative result for structurally nonzero occupancy as an invariant
failure. Do not replace this with an unconditional tolerance clamp.

Pre-slice evidence:

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
- Retain the global stable sort as the production default. Do not replace it
  wholesale with a k-way heap: the 2026-07-21 experiment improved sparse paths
  modestly but substantially regressed dense paths. If memory pressure or new
  workload evidence justifies a hybrid later, preserve the explicit
  `(time, path index, event index)` ordering and benchmark the crossover.
- Let an `AcceptedChainState` build its occupancy index once and derive
  `pair_overlap`/`action` from that same index. Keep the independent full sweep as
  an audit/reference implementation.
- Cache at most the overlap (with its ledger generation) and derive action from
  immutable `U` on access. Use compensated delta accumulation or a configurable
  periodic rebase from the ledger for long chains, with the full evaluator as an
  independent drift check.
- Completed 2026-07-21: add a pure
  `measure_interaction(configuration, model)` bundle so non-sampler callers can
  obtain all interaction observables with one full sweep, while the sampler
  uses the same calculation with its accepted-state overlap cache.
- Prune empty timelines or use a flat hash map keyed by `SiteId`.
- Either keep the arbitrary-path full evaluator private as an explicitly named
  test oracle or exercise it as a real production fallback; remove the unused
  private header/include and outdated proposal comment in either case.

The independent evaluator remains valuable for tests; sharing site reduction and
event transition semantics does not require sharing the whole algorithm.
The k-way/global-sort split proposed by the porting guide
(`../docs/CPP_PORT.md:414-431`) was tested and is not the current recommendation
without a workload-specific crossover.

### 10. P1: make permutation topology authoritative

Status (2026-07-21): the continuous and retained ideal slices are complete. A
validated `Permutation` now owns authoritative successors and a private
deterministic cycle cache, exposing both only through read-only views.
`ContinuousConfiguration` stores one private topology value instead of parallel
public cycle and successor vectors; its path storage is also private as of the
valid-by-construction configuration slice. Sampling, time rotation, cycle
updates, and stitch construction consume that value. Stitch preparation
validates the complete proposed permutation while the occupancy replacement is
staged, and acceptance publishes topology with one non-throwing move.
Construction, deterministic decomposition, invalid-successor,
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

Status (2026-07-21): the option-value and prevalidation slice is complete.
`SegmentUpdateOptions`, `StitchUpdateOptions`, `StitchSweepOptions`, and
`RandomSeamStitchOptions` replace the positional update signatures. Pair and
collective proposals now enter through one `stitch_update()` selected by an
explicit strand count, with an optional anchor or complete explicit strand set.
All option fields are validated before randomness, including fields unused by
an explicit interval or zero update count. `sweep()` builds one private prepared
plan before its first move, and `run()` reuses that plan and its prepared stitch
mixture across burn-in and sampling. The random-seam macro-kernel likewise
prepares its fixed-seam request before the first rotation. Clone-based
regressions prove invalid sweep, run, and random-seam requests preserve the
configuration, occupancy-derived caches, statistics, and subsequent random
stream. Valid default segment/stitch updates remain no-ops for undersized
systems, while explicitly malformed segment/stitch options now fail
consistently. The legacy two-strand matching order is also removed: every
supported strand count now evaluates its matrix in row-major order and samples
through the same validated `PreparedPermanent` recursion. An exact 2x2
permanent regression and its sampled identity/exchange law cover the unified
path; seeded pair-stitch streams may change while the proposal law remains
unchanged. Translation-unit splitting, seam/bridge-distribution caching, and
fixed-capacity strand selection remain open.

Pre-slice evidence:

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

Remaining recommendation:

- Move pure topology/geometry selection into `stitch_selection.cpp` and bridge
  matching/splicing into `stitch_proposal.cpp`; keep sampler orchestration and
  commit in `interacting_sampler.cpp`.
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
- Completed 2026-07-21: route `k == 2` through the same prepared-permanent
  evaluator and sampler as all other supported sizes. The exact permanent and
  sampled pair law are covered directly; its seeded sample sequence may change.

A release-demo diagnostic using `U=0`, `beta=1`, `t=1`, `N=16`, `L=4`, `d=2`
and 5,000 random-seam macro-steps with one stitch attempt per step took
0.22-0.23 s for `k=2`, 0.35-0.36 s for `k=4`, and 0.82-0.87 s for `k=8` across
three runs. The macro-step also includes two time rotations and trace output, so
this does not isolate permanent or bridge cost; it is a compact regression
workload for the proposed seam and bridge-distribution caches.

### 12. P2: improve sampling workflow APIs

Status (2026-07-21): the prepared-plan and early-validation slice is complete.
`sweep()` validates every move option, including inactive branches, before the
first move or random draw. `run()` prepares that same plan once before burn-in
and reuses it for every sweep. Random-seam stitch options are validated before
the first time rotation. Invalid-plan regressions compare a copied sampler
before and after the failure and after a subsequent valid plan, covering state,
caches, statistics, and RNG progression. Streaming observers, an integrated
macro-kernel plan, statistics helpers, copy policy, checkpoints, and trace
metadata remain open.

Remaining evidence:

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
   retained-grid provenance completed 2026-07-20. Shared validation-helper
   cleanup and a general lattice-field abstraction remain.
4. Completed 2026-07-20: `PathCursor`/`PathSlice` migration for split,
   resample, stitch, and time-origin rotation, with exact traversal and boundary
   equivalence tests.
5. Completed 2026-07-21: `ContinuousPath`, retained dense/configuration,
   `ContinuousConfiguration`, and `Model` encapsulation, plus authoritative
   continuous and retained `Permutation` topology; full path/configuration
   validation remains available as an explicit diagnostic audit.
6. Retained measurement context completed 2026-07-20 and equal-time/density
   accumulators completed 2026-07-21; add cycle/winding/geometry accumulators,
   then optimize direct correlation kernels from benchmark evidence.
7. Log-weight categorical draws and adaptive-support numerics completed
   2026-07-20; canonical derivative and partition recurrences,
   spectrum/trigonometric caching, reusable free-path kernels, and cycle-label
   sampling and occupation-moment scratch reuse completed 2026-07-21.
8. Accepted-chain ownership completed 2026-07-20; a production k-way event
   merge was benchmarked and reverted 2026-07-21. Add the bundled interaction
   measurement separately.
9. Segment/stitch option values and early compound-plan preparation completed
   2026-07-21; split stitch selection/proposal code and add seam caches
   separately.
10. Split translation units and demos after responsibilities have stabilized.
11. GoogleTest exclusion from clang-tidy completed 2026-07-19; add install, CI,
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
