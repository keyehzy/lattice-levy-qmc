# C++ porting guide

The Python code is compact enough to serve as the behavioral specification for
a C++ rewrite. This guide proposes component boundaries and interfaces while
calling out semantics that must survive the change of language. It does not
require the C++ implementation to reproduce NumPy's memory layout or random
stream byte for byte.

## 1. Porting contract

A faithful implementation should preserve:

- the canonical, fixed-`N` target measure;
- the exact finite-torus one-particle trace and cycle recursion;
- explicit covering-space winding sectors;
- conditioned free bridges with a certified `tail_tolerance`;
- event-driven continuous paths with no Trotter grid in the finite-`U`
  sampler;
- right-continuous event semantics and the `(tau0, tau1]` resampling interval;
- permutation/cycle endpoint invariants;
- proposal kernels whose free weights cancel, leaving the interaction-only
  Metropolis ratio;
- the current observable definitions and units.

The port may deliberately change internal storage, numerical algorithms, or
RNG implementation if tests demonstrate the same distributions and
invariants.

## 2. Recommended source boundaries

```text
include/levy/
  model.hpp             validated parameters and numerical options
  random.hpp            all stochastic primitives behind one RNG object
  free_numerics.hpp     log-sum-exp, Bessel count and winding inversion
  free_boson.hpp        torus trace, canonical table, cycle sampling
  path.hpp              sites, events, continuous paths, bridge operations
  configuration.hpp     ideal and continuous configuration records
  interaction.hpp       overlap/action and observable estimators
  sampler.hpp           move statistics and finite-U Markov chain
src/
  ...matching implementation files...
tests/
  ...unit, invariant, statistical, and ED regression tests...
```

This removes the current cross-module import of the private
`_sample_bessel_pair_count`: both midpoint and continuous bridge code should
depend on the same `free_numerics` primitive.

Keep model parameters separate from mutable chain state. The Python records
repeat `L`, `d`, `N`, `beta`, and `t` for convenience; a C++ configuration can
hold a shared immutable `Model` or be owned by a sampler that supplies the
model to free functions.

## 3. Core types

The following declarations are illustrative interfaces, not mandated syntax:

```cpp
namespace levy {

using Coord = std::int64_t;
using ParticleId = std::uint32_t;
using Axis = std::uint32_t;
using Site = std::vector<Coord>;        // always model.dimension entries
using Cycle = std::vector<ParticleId>; // directed cyclic label order

struct Model {
  std::size_t particle_count;
  double beta;
  Coord linear_size;
  std::size_t dimension;
  double hopping;
  double interaction;

  void validate_free() const;        // beta >= 0
  void validate_interacting() const; // beta > 0
  std::size_t volume() const;         // checked integer exponentiation
};

struct NumericalOptions {
  double tail_tolerance = 1e-14;
  std::size_t max_bessel_terms = 2'000'000;
  std::size_t max_winding = 1'000'000;
};

struct JumpEvent {
  double time;
  Axis axis;
  std::int8_t direction; // exactly -1 or +1
};

struct ContinuousPath {
  double duration;
  Site start;
  Site end;
  std::vector<JumpEvent> events; // sorted, internal event times

  void validate(std::size_t dimension) const;
  Site position_at(double tau) const;
  std::size_t event_count() const noexcept;
};

struct FreeBosonTable {
  std::vector<double> log_z; // index 0 unused/-infinity
  std::vector<double> log_Z; // log_Z[0] == 0
};

struct ContinuousConfiguration {
  std::vector<Cycle> cycles;
  std::vector<ParticleId> permutation;
  std::vector<ContinuousPath> worldlines; // indexed by particle label
  double log_Z0_N;

  void validate(const Model&) const;
  std::size_t event_count() const noexcept;
  std::vector<Coord> total_winding(const Model&) const;
};

struct MoveStatistics {
  std::uint64_t attempts = 0;
  std::uint64_t accepts = 0;
  std::optional<double> acceptance() const;
};

} // namespace levy
```

Python stores every jump as a dense `d`-component vector. `JumpEvent` stores
only axis and sign, reducing event memory from `O(E*d)` to `O(E)` and making
the nearest-neighbor invariant explicit. A structure-of-arrays representation
is also reasonable if profiling favors it.

For the ideal skeleton, use a flat row-major buffer rather than nested vectors:

```cpp
struct DenseWorldlines {
  std::size_t particles;
  std::size_t time_points; // M + 1
  std::size_t dimension;
  std::vector<Coord> values;

  Coord& at(ParticleId p, std::size_t m, Axis a);
};

struct IdealCyclePath {
  Cycle labels;
  Site base_point;
  Site winding;
  std::vector<Site> covering_path; // ell*M + 1 points
  std::vector<Site> torus_path;    // same path reduced modulo L
};

struct IdealSkeleton {
  std::size_t time_links_per_beta; // M
  std::vector<IdealCyclePath> cycles;
  std::vector<ParticleId> permutation;
  DenseWorldlines worldlines;
  DenseWorldlines worldlines_covering;
  double log_ZN;

  void validate(const Model&) const;
};
```

Keep covering and torus buffers distinct, as the Python `IdealBosonConfiguration`
does. Winding cannot be reconstructed from already reduced coordinates.

## 4. Numerical and free-boson interfaces

An explicit RNG wrapper makes ownership and test injection clear:

```cpp
class Random {
 public:
  explicit Random(std::uint64_t seed);
  double uniform_open();
  std::uint64_t uniform_index(std::uint64_t upper_exclusive);
  std::uint64_t binomial(std::uint64_t n, double p);
  std::size_t discrete_index(std::span<const double> probabilities);
  void shuffle(std::span<ParticleId> labels);
};
```

Every stochastic API should receive `Random&`; avoid hidden thread-local or
global engines. The current Python implementation is reproducible through one
explicit `numpy.random.Generator`. Exact cross-language sample streams are not
a useful default requirement because standard-library distribution algorithms
are implementation dependent.

Recommended free-function surface:

```cpp
std::uint64_t sample_bessel_pair_count(
    std::uint64_t abs_delta, double lambda, Random&,
    const NumericalOptions&);

Coord sample_midpoint_covering_1d(
    Coord a, Coord b, double tau_left, double tau_right,
    double hopping, Random&, const NumericalOptions&);

std::vector<Site> sample_bridge_covering(
    const Site& a, const Site& b, double duration, std::size_t steps,
    double hopping, Random&, const NumericalOptions&);

double log_one_particle_trace(double duration, const Model&);
FreeBosonTable canonical_table(const Model&);
std::vector<Cycle> sample_cycle_labels(
    const FreeBosonTable&, std::size_t particle_count, Random&);
Coord sample_winding_1d(
    Coord linear_size, double duration, double hopping,
    Random&, const NumericalOptions&);

ContinuousPath sample_continuous_bridge(
    const Site& a, const Site& b, double duration, double hopping,
    Random&, const NumericalOptions&);

std::vector<ContinuousPath> split_path(
    const ContinuousPath&, std::span<const double> cut_times);

ContinuousPath resample_interval(
    const ContinuousPath&, double tau0, double tau1, double hopping,
    Random&, const NumericalOptions&);
```

The model-level functions can be wrapped by stateful classes later if
profiling shows a benefit from cached momentum grids or Bessel workspaces.
Starting with pure functions keeps their probability contracts testable.

## 5. Sampler interface

```cpp
struct Observables {
  double action;
  double pair_overlap_time;
  double double_occupancy_per_site;
  double kinetic_energy;
  double interaction_energy;
  double total_energy;
  std::size_t event_count;
  std::vector<Coord> winding;
  std::vector<std::size_t> cycle_lengths;
};

struct SweepOptions {
  std::optional<std::size_t> segment_updates; // null => N
  double segment_fraction = 0.25;
  std::size_t cycle_updates = 1;
  std::size_t global_updates = 0;
};

class InteractingSampler {
 public:
  InteractingSampler(Model, NumericalOptions, std::uint64_t seed);

  bool segment_update(
      std::optional<ParticleId> particle = {},
      std::optional<std::pair<double, double>> interval = {},
      double fraction = 0.25);
  bool whole_worldline_update(std::optional<ParticleId> particle = {});
  bool cycle_update(std::optional<std::size_t> cycle_index = {});
  bool global_update();
  void sweep(const SweepOptions& = {});

  Observables observables() const;
  const ContinuousConfiguration& state() const noexcept;
  const std::unordered_map<std::string, MoveStatistics>& statistics() const;

 private:
  Model model_;
  NumericalOptions numerical_;
  Random random_;
  ContinuousConfiguration state_;
  double pair_overlap_;
  double action_;
};
```

A library interface should not return a `vector<map<string, object>>` like the
dynamic Python `run()` method. Prefer `Observables`, a callback, or an output
iterator so a long run can stream measurements without retaining all of them:

```cpp
sampler.run(run_options, [](std::size_t sample, const Observables& value) {
  // accumulate, serialize, or send to an analysis pipeline
});
```

## 6. Python-to-C++ mapping

| Python symbol | Proposed C++ owner | Notes |
| --- | --- | --- |
| `_sample_bessel_pair_count` | `free_numerics` | Shared primitive, no longer private to the grid sampler |
| `sample_midpoint_covering*`, `sample_bridge_covering` | `free_numerics` / free path API | Dense observation-grid output |
| `periodic_kernel_scaled_1d` | `free_numerics` | Optional if direct torus midpoint sampling remains public |
| `canonical_log_partition` | `free_boson` | Return `FreeBosonTable`; cache per immutable model |
| `sample_cycle_labels`, `sample_winding_1d` | `free_boson` | Topology and winding draws |
| `CyclePath`, `IdealBosonConfiguration` | `configuration` | Dense ideal-only records |
| `ContinuousPath` | `path` | Sparse immutable/value-like path |
| `ContinuousConfiguration` | `configuration` | Mutable accepted Markov state |
| `_paths_for_cycle` | `free_boson` internal helper | Returns label/path associations or fills indexed paths |
| `pair_overlap_time`, estimator functions | `interaction` | Pure functions over model and configuration |
| `_metropolis_accept` | `sampler` internal helper | Compare logarithms; do not form `exp(-delta)` |
| `InteractingLatticeLevySampler` | `InteractingSampler` | Own model, RNG, accepted state, caches, and counters |

## 7. Language-level traps

### Euclidean modulo

Python/NumPy reduction by positive `L` always returns a value in `[0, L)`.
C++ `%` leaves a negative remainder for a negative coordinate. All physical
site keys and endpoint checks need a helper such as:

```cpp
Coord torus_mod(Coord x, Coord L) {
  const Coord r = x % L;
  return r < 0 ? r + L : r;
}
```

Do not reduce covering coordinates in place; doing so destroys winding.

### Event boundaries

`position_at(tau)` includes all events with `event.time <= tau`, equivalent to
`upper_bound`. Interval replacement retains old events at `time <= tau0`,
replaces events in `(tau0, tau1]`, and retains old events at `time > tau1`.
Sampled event times must be strictly internal to the bridge duration.

The overlap sweep groups events with exactly equal floating-point times and
integrates no duration between them. A stable sort retains deterministic
ordering inside such a group, although the ordering cannot change the
integrated action.

### Integer widths and checked arithmetic

Use signed 64-bit covering coordinates. Check conversions and products such as
`L*winding`, Bessel order `L*abs(w)`, `L**d`, and the event-count sum. Particle
labels and vector indices are nonnegative, but do not silently mix `size_t`
with signed coordinate arithmetic.

### Immutability and rejected proposals

Python can restore rejected moves cheaply because `ContinuousPath` is frozen
and configurations hold path references. In C++, either:

- build replacements off-state, compute a delta or trial action, and move them
  into the state only after acceptance; or
- use a small rollback guard that owns the displaced paths until commit.

Avoid leaving cached `pair_overlap_` or `action_` inconsistent if allocation or
action evaluation throws. The invariant is always
`action_ == model.interaction * pair_overlap_` for the accepted state.

### Empty systems and zero parameters

Preserve explicitly tested boundary behavior:

- `N == 0` is valid;
- the interacting model requires `beta > 0`;
- the ideal mathematical helpers allow nonnegative durations;
- `t == 0` permits only zero-displacement bridges and yields zero winding;
- `U` may be any finite value;
- a no-particle segment or cycle update returns success without fabricating a
  path.

Define whether no-op calls increment statistics and test that decision. The
Python segment/cycle early returns do not increment attempts.

## 8. Special functions and stability

The port needs equivalents for:

- `lgamma`, available as `std::lgamma`;
- log-sum-exp over short vectors;
- exponentially scaled modified Bessel \(I_n(z)e^{-|z|}\);
- binomial, uniform, discrete, and without-replacement sampling.

`std::cyl_bessel_i` is not a direct replacement for SciPy `ive`: it can
overflow before the common exponential factor is removed. Select a numerical
library with a scaled Bessel function, or implement/test a stable recurrence or
log-domain evaluator. This affects winding probabilities and the exact-PMF
test helper.

The Bessel pair-count sampler itself uses `lgamma` and log weights, plus a
geometric tail proof. Preserve the distinction between:

- the configured relative omitted-mass bound;
- underflow of already negligible weights;
- a hard work limit (`max_bessel_terms` or `max_winding`), which should raise a
  descriptive error rather than silently bias a draw.

The Metropolis decision should compare `log(uniform)` with `-delta_action`.
There is no need to reproduce Python's literal `745.0` cutoff if the log-space
implementation handles the range safely and has the same acceptance law.

## 9. Performance plan

First reproduce behavior, then optimize measured bottlenecks. Likely wins are:

1. cache `FreeBosonTable` and momentum cosines for an immutable model;
2. use axis/sign events instead of dense jump vectors;
3. reserve event buffers from sampled jump counts before filling them;
4. merge already sorted per-path event streams for the action instead of
   constructing and sorting one global event vector;
5. compute local overlap deltas for segment/cycle proposals;
6. replace list-based remaining-label deletion with an indexed pool if large
   `N` makes cycle sampling material;
7. run independent chains in parallel, keeping one RNG and sampler state per
   thread.

Items 4 and 5 change the most delicate correctness path. Keep the full
`O(E log E)` action evaluator as a debug/reference implementation and compare
incremental results against it in randomized tests.

## 10. Incremental implementation order

1. **Numerical foundation:** validation, Euclidean modulo, log-sum-exp,
   conditioned Bessel counts, and scaled Bessel support.
2. **Free grid sampler:** one-dimensional midpoint, multidimensional bridge,
   torus trace, and winding.
3. **Canonical topology:** partition table and labeled cycle sampling.
4. **Ideal skeleton:** dense covering/torus paths and permutation assembly.
5. **Continuous paths:** event bridge, position lookup, split, and interval
   replacement.
6. **Configuration/action:** invariants, total winding, exact overlap sweep,
   and estimators.
7. **Finite-`U` sampler:** segment, cycle, and global proposals with cache-safe
   acceptance.
8. **Drivers and serialization:** demos, streamed measurements, diagnostics,
   and optional checkpointing.

Each stage has a direct Python oracle before the next layer is introduced.

## 11. Verification strategy

Port deterministic identities before statistical tests:

| Test | What it protects |
| --- | --- |
| Bessel convolution | Kernel and bridge normalization |
| Torus trace vs direct momentum sum | Finite-volume convention and factors of `d` |
| Canonical recursion vs permutation enumeration | Bosonic combinatorics |
| Hand-built path positions and splitting | Right continuity and interval ownership |
| Hand-built overlap integral | Torus modulo, occupancy updates, and time integration |
| Configuration validation | Cycle/permutation/endpoint invariants |
| `U=0` acceptance | Cancellation of free proposal weights |
| Cached action vs full recomputation | Transactional move correctness |

Then add distributional tests with fixed statistical criteria:

- midpoint histograms against the exact Bessel PMF;
- winding histograms against enumerated scaled-Bessel weights;
- cycle-length frequencies against canonical probabilities;
- ideal event-count and observable moments against the Python implementation;
- small finite-`U` observables against exact diagonalization.

Do not compare individual seeded configurations unless the port intentionally
implements NumPy's bit generator and every distribution transform. Compare
laws, exact identities, and state invariants instead. Run the statistical tests
with enough samples to detect meaningful bias and thresholds loose enough to
avoid flaky failures; keep expensive ED/statistical validation separate from
fast unit tests if necessary.

## 12. Decisions to make before freezing the C++ API

- Numerical backend for scaled Bessel functions.
- Header-only templates versus a compiled library and stable ABI.
- Error policy: exceptions, expected/error values, or a mixed boundary.
- Whether `Model` owns interaction `U` or separates free parameters from the
  target action for easier reuse.
- Whether paths remain value objects or use shared immutable storage for cheap
  proposals and snapshots.
- Measurement streaming and checkpoint format.
- Threading model and deterministic per-chain seed derivation.

None of these decisions changes the mathematical interfaces above. They should
be settled with the intended deployment environment and profiling data rather
than encoded accidentally while translating Python line by line.
