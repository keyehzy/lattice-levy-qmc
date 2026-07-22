# Event-based continuous-time measurements

Date: 2026-07-22

Status: implementation in progress. The shared Matsubara mode/result-shape
layer, retained-result migration, and continuous phase plan (step 1), plus the
owning continuous measurement context and deterministic event-sweep boundary
tests (step 2), plus the combined density/flux primitive and its deterministic
identity tests (step 3), were completed on 2026-07-22. This document remains
the design specification and contains no implementation.

## Scope and recommendation

The interacting sampler should gain a measurement layer built around exact
projections of its piecewise-constant paths and hopping events onto selected
Matsubara modes. It should not acquire a retained imaginary-time grid, and the
measurements should not become methods or callbacks owned by
`InteractingSampler`.

The clean abstraction is an **additive continuous-time signal**. For one
configuration, one component of a general supported observable has the mode
amplitude

\[
X_n =
\sum_{[a,b)} x(S_{[a,b)})
       \int_a^b e^{i\omega_n\tau}\,d\tau
+ \sum_g y(G_g,S_{g^-},S_{g^+})e^{i\omega_n\tau_g}.
\]

The first term is a residence/segment contribution from a diagonal observable
of the constant occupation state. The second is an impulse contribution from a
zero-duration group `G_g` of coincident jump events. A geometric impulse such
as flux is the sum of its individual hop contributions inside the group; an
impulse that depends on global occupation must define group-level pre/post
semantics and may not observe order-dependent transient states. Density is a
residence signal; signed hopping flux is an impulse signal; other observables
can use either or both. Spatial Fourier factors are part of `x` or `y`.

This mathematical abstraction should guide a small composition of concrete
types, not a public `Observable` base class:

1. `ContinuousMeasurementContext` owns the physical event timeline derived
   once from a `ContinuousConfiguration`.
2. `MatsubaraModeSet` owns validated Fourier geometry, while
   `ContinuousMatsubaraPlan` adds reusable phase data.
3. A per-configuration projector produces unnormalised density and hopping-flux
   mode amplitudes in one event pass.
4. Typed accumulators own ensemble statistics and return shape-safe,
   convention-bearing results.

This is consistent with the repository's existing
`RetainedMeasurementContext`/typed-accumulator pattern while respecting that a
`ContinuousPath` already contains the exact jump times. It also leaves a clear
extension seam for diagonal local observables, event observables, and mixed
correlations without committing the public API to a callback hierarchy before
there are several real consumers.

## Current-state evidence reviewed

This design was checked against commit `ef114cb` and the user-supplied
[`conversation.txt`](../conversation.txt). The most relevant implementation
boundaries are:

- [`path.hpp`](../cpp/include/qmc/path.hpp) and
  [`path.cpp`](../cpp/src/path.cpp): valid-by-construction event paths,
  right-continuity, and covering directions;
- [`continuous_configuration.hpp`](../cpp/include/qmc/continuous_configuration.hpp)
  and
  [`continuous_configuration.cpp`](../cpp/src/continuous_configuration.cpp):
  model/topology ownership, physical time rotation, and winding;
- [`interaction.cpp`](../cpp/src/interaction.cpp): the existing full timed-event
  collection, equal-time grouping, occupancy replay, and pair-overlap integral;
- [`path_cursor.hpp`](../cpp/src/path_cursor.hpp): the distinct per-path
  cut/slice abstraction and its endpoint-side semantics;
- [`observables.hpp`](../cpp/include/qmc/observables.hpp),
  [`retained_observables.cpp`](../cpp/src/retained_observables.cpp), and
  [`lattice_transforms.cpp`](../cpp/src/lattice_transforms.cpp): retained
  context/accumulator conventions, shape ownership, and the existing Fourier
  signs/normalisation;
- [`interacting_sampler.hpp`](../cpp/include/qmc/interacting_sampler.hpp) and
  [`interacting_demo.cpp`](../cpp/examples/interacting_demo.cpp): the current
  scalar `run()` boundary and explicit sampling loop; and
- [`AGENTS.md`](../cpp/AGENTS.md): the requirements to preserve continuous and
  retained representations, prefer composition over frameworks, validate at
  public boundaries, and document units and endpoint conventions.

## Requirements derived from the repository

The design needs to preserve the following properties.

- `ContinuousConfiguration` and `ContinuousPath` are already valid by
  construction. Measurement code should consume their read-only views and not
  repeat full structural validation.
- Positions are right-continuous. Events may occur at `0` or `beta`, and equal
  event times are legal. Boundary rules must have one implementation.
- Covering-space coordinates remain authoritative for winding, while density
  and bond phases use physical torus geometry.
- `M` belongs only to retained ideal samples. No continuous-time public type
  should contain a retained-time count or silently introduce a quadrature grid.
- Public inputs and flat result extents must be validated and checked for
  overflow before allocation or accumulator mutation.
- Repeated work should be bound to an immutable owner. A Matsubara plan is
  reused across configurations; a measurement context is reused across
  observables for one configuration.
- Accumulators validate provenance before mutation, own their sample count, and
  normalise only in `finish()`, following the current retained accumulators.
- The implementation belongs to the `qmc_interacting` target because it depends
  on `ContinuousConfiguration`, but public names belong to namespace `qmc`.
  There is no `qmc::interacting` namespace in the current codebase.
- Measurement orchestration remains outside `InteractingSampler`. This avoids
  growing `run()`, which currently materialises scalar `InteractingObservables`,
  and remains compatible with the separate
  [planned streaming-run work](https://github.com/keyehzy/lattice-levy-qmc/issues/5).

The current code also points to two useful implementation extractions:

- `interaction.cpp` privately collects, stable-sorts, groups, and applies timed
  events while maintaining physical positions and occupancies.
- `PathCursor` centralises cut-side semantics for path surgery, but is a
  per-path monotone-cut abstraction rather than a global configuration event
  sweep.

The new layer should reuse a global event-sweep utility with the interaction
evaluator eventually; it should not stretch `PathCursor` into a second,
unrelated responsibility.

### Refinements to the conversation's initial sketch

The conversation has the right primary representation and estimators, with a
few changes needed for the current codebase.

- Use namespace `qmc` and the `qmc_interacting` target, not a new
  `qmc::interacting` namespace.
- Keep event geometry in `ContinuousMeasurementContext` and reusable Fourier
  work in `ContinuousMatsubaraPlan`. Constructing the context with one density/
  current spec would make it less reusable for non-Fourier event observables.
- Keep a geometry-only mode descriptor separate from complete sample `Model`
  provenance so retained and continuous transforms can share shapes without
  weakening accumulator compatibility checks.
- Call the event object `flux` and its covariance the full gauge response.
  Return the diamagnetic and derived paramagnetic pieces explicitly.
- For this homogeneous fixed-`N`, zero-field model, centre density and flux by
  their exact symmetry means. Retain online complex co-moments as the general
  mechanism for a future observable whose mean is not known analytically.
- Treat a requested-lag imaginary-time result as a separate exact backend, not
  as the internal representation of the Matsubara measurement.

## Proposed ownership model

### `MatsubaraModeSet` and `ContinuousMatsubaraPlan`

A mode set and its plan should be constructed once for a run from the free
model's `beta` and `TorusLayout` and a request such as:

```cpp
struct MatsubaraModeRequest {
  // Each momentum is an integer vector k with q_alpha = 2*pi*k_alpha/L.
  std::vector<std::vector<std::size_t>> momentum_indices;
  // Signed n in omega_n = 2*pi*n/beta.
  std::vector<std::int64_t> frequency_indices;
};

class MatsubaraModeSet {
public:
  MatsubaraModeSet(double beta, TorusLayout layout,
                   MatsubaraModeRequest request);

  [[nodiscard]] double beta() const noexcept;
  [[nodiscard]] const TorusLayout &layout() const noexcept;
  [[nodiscard]] std::size_t momentum_count() const noexcept;
  [[nodiscard]] std::size_t frequency_count() const noexcept;
  [[nodiscard]] std::size_t mode_count() const noexcept;
  [[nodiscard]] std::span<const std::size_t>
  momentum_indices(std::size_t momentum) const;
  [[nodiscard]] double wavevector_component(std::size_t momentum,
                                             std::size_t axis) const;
  [[nodiscard]] std::int64_t frequency_index(std::size_t frequency) const;
  [[nodiscard]] double frequency(std::size_t frequency) const;
  bool operator==(const MatsubaraModeSet &) const = default;
};

class ContinuousMatsubaraPlan {
public:
  // Continuous-time phase reduction is supported for |n| <= this value.
  static constexpr std::int64_t kMaximumAbsoluteFrequencyIndex = 1'048'576;

  explicit ContinuousMatsubaraPlan(MatsubaraModeSet modes);

  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  // Private reusable phase data.
};

template <class T> class MatsubaraModeField {
public:
  // Requires exactly modes.mode_count() values.
  MatsubaraModeField(MatsubaraModeSet modes, std::vector<T> values);

  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  [[nodiscard]] std::span<const T> values() const noexcept;
  [[nodiscard]] const T &at(std::size_t frequency,
                            std::size_t momentum) const;
};
```

The mode set should:

- require positive `beta` and validate every momentum component against its
  `TorusLayout`;
- require at least one momentum and one frequency;
- accept signed frequency indices so conjugation and Ward tests do not require
  a separate convention;
- reject duplicate requested rows and duplicate frequency indices rather than
  calculate indistinguishable output entries;
- preserve request order;
- check the Cartesian-product and component extents before allocating;
- define frequency-major, then request-order momentum storage; vector/tensor
  component axes follow after the mode index, with the right axis varying
  fastest for a response tensor;
- expose physical frequencies and wavevector components as derived values,
  while retaining integer indices as the authoritative exact mode identity.

`MatsubaraModeSet` is a geometry descriptor, so it does not impose the
continuous projector's stricter phase-accuracy limit. It rejects a frequency
only when its derived physical value is nonfinite or a nonzero index maps to
zero in `double`. `ContinuousMatsubaraPlan` additionally rejects every index with
`abs(n) > kMaximumAbsoluteFrequencyIndex`, computing the magnitude without
negating a possibly minimum-valued `int64_t`. Keeping this limit on the plan
allows a retained transform to describe every DFT row representable by the
signed index type without making a large retained `M` part of the
continuous-time numerical contract.

Momentum components are canonical in `[0,L)`; the component of `-q` is
`(L-k) % L`. Frequency indices remain signed rather than being reduced to a
finite time-grid range.

Do not reuse `SiteId` as a momentum identifier: its public contract says it is
one physical site produced by a `TorusLayout`. Integer component rows make the
reciprocal meaning explicit. A future `MomentumId` is reasonable if the shared
transform work demonstrates enough flat-momentum consumers to justify a second
strong index.

`MatsubaraModeSet` is deliberately geometry-only so the retained and exact
continuous transforms can share one result shape and Fourier convention.
`ContinuousParticleModes` and both continuous accumulators additionally own
and compare the complete free `Model`; sharing a Fourier shape is not
permission to mix different particle counts or hopping laws.

The on-site interaction is not part of `ContinuousConfiguration`, and neither
density nor flux geometry needs it. Mixing different `U` chains therefore
remains run-level provenance, just as it is for the existing scalar trace. A
future trace/checkpoint schema should record the `InteractingModel` alongside
the plan and result.

Separating the small immutable descriptor from the numerical plan lets result
objects own exact Fourier provenance without copying a potentially large phase
cache or borrowing a plan that may be destroyed. `ContinuousParticleModes` and
finished results own a `MatsubaraModeSet`; the primitive sample and accumulators
also own the full free `Model`; repeated projection consumes a retained
`ContinuousMatsubaraPlan`.

The plan can precompute momentum components and the
per-axis factors `exp(-i*q_alpha)` and `exp(-i*s*q_alpha/2)`. A full
`momentum_count * volume` site-phase table may be worthwhile for repeated
samples, but should be selected by a benchmark and a checked memory policy. It
should later share the transform workspace contemplated for retained
observables rather than creating a competing Fourier convention.

All plan allocation and validation happens in its constructor. Projection is
`const`, uses caller-owned temporaries, and has no lazy mutable cache or hidden
RNG, so one plan can be shared by independent measurement workers.

### `ContinuousMeasurementContext`: one configuration's event geometry

The context should own derived data rather than borrow from a mutable sampler
state. That matches `RetainedMeasurementContext` and lets a caller safely use
the context after the sampler advances.

Conceptually it contains:

```cpp
struct ContinuousHop {
  double time;
  ParticleId particle;
  SiteId departure;
  SiteId arrival;
  Axis axis;
  std::int8_t direction;

  bool operator==(const ContinuousHop &) const = default;
};

class ContinuousMeasurementContext {
public:
  explicit ContinuousMeasurementContext(
      const ContinuousConfiguration &configuration);

  [[nodiscard]] const Model &model() const noexcept;
  [[nodiscard]] const TorusLayout &layout() const noexcept;
  // Physical positions immediately before events at the time-zero seam.
  [[nodiscard]] std::span<const SiteId> seam_positions() const noexcept;
  // Nondecreasing time order; ties retain particle/path event order.
  [[nodiscard]] std::span<const ContinuousHop> hops() const noexcept;
  [[nodiscard]] std::size_t event_group_count() const noexcept;
  [[nodiscard]] double event_time(std::size_t group) const;
  [[nodiscard]] std::span<const ContinuousHop>
  hops_at(std::size_t group) const;
};
```

Construction traverses each path once, derives each physical departure and
arrival without losing the covering-space direction, and globally orders the
hops. Equal-time hops are one zero-duration group. The interval preceding a
group sees the state after the previous group; every hop in the group is then
applied before the next positive-duration interval. This reproduces the
current pair-overlap semantics.

`seam_positions()` deliberately means the state before time-zero events.
Right-continuous values at `tau == 0` are obtained by applying that group. This
distinction is necessary because `ContinuousPath::start()` is the pre-event
endpoint while `position_at(0)` includes events at zero. Events at `beta` are
applied after the final positive-duration interval and close the permutation
loops at the seam.

The initial implementation should use the same time normalisation and stable
tie ordering as `interaction.cpp`. A later migration can make a private
`ContinuousEventSweep` the common implementation for the context and full
pair-overlap calculation. Collection plus `stable_sort` is the correctness
baseline. A k-way merge of already sorted path event lists is only justified
after measurement workloads show the sort to matter.

Intervals end at `context.model().beta()`, not at a separately supplied value.
The existing few-ulp worldline-duration tolerance and event-at-`beta`
normalisation must match the interaction sweep exactly, including paths whose
stored duration lies just below or above the model value.

The group accessors keep coincident-event semantics in one implementation; a
projector should not rediscover groups with its own floating-point comparison.
The public context is useful for additional projectors, but it should not expose
mutable positions or occupancies. A private sweep state can replay the owned
hops with particle positions and a sparse occupancy map when an observable
needs them.

The private sweep contract should make its atomicity explicit. Conceptually,
each group is processed as follows:

```text
emit residence interval [previous_time, group_time) from the current state
snapshot the group-level pre-state
derive the post-state by applying every hop in stable path order
emit group impulses from (hops, pre-state, post-state)
install the post-state and continue
```

After the last group, it emits `[previous_time, beta)`. A group at zero thus
precedes every positive-duration interval; a group at `beta` follows the final
interval but still contributes periodic event impulses. Applying individual
hops in a stable order is necessary to derive the final state and each hop's
geometric departure, but no occupation-dependent observable may see the
order-dependent intermediate states inside a zero-duration group. Concrete
projectors can specialise away pre/post snapshots when they need only cached
residence values or geometric hop data.

Events stored at `0` and `beta` have the same Matsubara phase but remain two
ordered seam-side groups, matching the path's chosen cut and endpoint
semantics. The initial density and geometric-flux projectors are insensitive to
that ordering. A future state-dependent impulse that wants to treat both sides
as one cyclic coincident insertion must first define its operator ordering and
permutation-seam replay; it must not silently coalesce the two groups.

This is the reusable implementation seam: interval integration, atomic event
groups, and replay state are shared; the functions mapping a state to a
residence value or a group to impulses remain observable-specific. It is
deliberately an internal contract until multiple concrete projectors establish
whether a template, a small state machine, or direct loops give the clearest
code.

### Per-configuration conserved-particle modes

Density and hopping flux share phase conventions and an exact continuity
identity, so their primitive amplitudes should be calculated together:

```cpp
class ContinuousParticleModes {
public:
  [[nodiscard]] const Model &model() const noexcept;
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  [[nodiscard]] std::complex<double>
  density(std::size_t frequency, std::size_t momentum) const;
  [[nodiscard]] std::complex<double>
  flux(std::size_t frequency, std::size_t momentum,
       std::size_t axis) const;
  [[nodiscard]] std::size_t axis_event_count(std::size_t axis) const;
};

[[nodiscard]] ContinuousParticleModes
continuous_particle_modes(const ContinuousMeasurementContext &context,
                          const ContinuousMatsubaraPlan &plan);

// One-off convenience overload.
[[nodiscard]] ContinuousParticleModes
continuous_particle_modes(const ContinuousConfiguration &configuration,
                          const ContinuousMatsubaraPlan &plan);
```

The result owns its complete free `Model` and `MatsubaraModeSet` as values. The
mode descriptor follows the same grid-provenance pattern as
`ImaginaryTimeDensityCorrelations`, while the additional model prevents
cross-ensemble accumulation.

This primitive result is intentionally unnormalised. It is a deterministic
measurement of one configuration and is useful for exact identities and raw
traces. Ensemble normalisation and connected/response semantics belong to
typed accumulators.

An efficient global sweep keeps the current density mode `n_q` for every
requested momentum. Before each event group it adds `n_q` times the exact
interval transform for every requested frequency. Each hop then:

- adds its signed bond-midpoint impulse to the matching flux component; and
- updates `n_q` by the arrival phase minus the departure phase.

This calculates both primitive fields without calls to `positions_at(tau)` and
without a measurement quadrature.

The combined baseline intentionally keeps the continuity-related fields
together. If profiling later shows that density-only runs are common and flux
projection is material, add a concrete density-only projector over the same
context and plan; do not put optional, possibly absent arrays into the primitive
result.

### Typed ensemble accumulators

The initial public accumulators should be concrete:

```cpp
class ContinuousMatsubaraDensityCorrelations {
public:
  [[nodiscard]] const Model &model() const noexcept;
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  [[nodiscard]] std::size_t sample_count() const noexcept;
  [[nodiscard]] std::complex<double>
  mean_amplitude(std::size_t frequency, std::size_t momentum) const;
  [[nodiscard]] double at(std::size_t frequency,
                          std::size_t momentum) const;
};

class HoppingResponse;

class DensityMatsubaraAccumulator {
public:
  DensityMatsubaraAccumulator(Model, MatsubaraModeSet);
  [[nodiscard]] const Model &model() const noexcept;
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  [[nodiscard]] std::size_t sample_count() const noexcept;
  void observe(const ContinuousParticleModes &);
  [[nodiscard]] ContinuousMatsubaraDensityCorrelations finish() const;
};

class HoppingResponseAccumulator {
public:
  HoppingResponseAccumulator(Model, MatsubaraModeSet);
  [[nodiscard]] const Model &model() const noexcept;
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  [[nodiscard]] std::size_t sample_count() const noexcept;
  void observe(const ContinuousParticleModes &);
  [[nodiscard]] HoppingResponse finish() const;
};

class HoppingResponse {
public:
  [[nodiscard]] const Model &model() const noexcept;
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  [[nodiscard]] std::size_t sample_count() const noexcept;
  [[nodiscard]] std::complex<double>
  mean_flux(std::size_t frequency, std::size_t momentum,
            std::size_t axis) const;
  [[nodiscard]] std::complex<double>
  flux_response(std::size_t frequency, std::size_t momentum,
                std::size_t left, std::size_t right) const;
  [[nodiscard]] double diamagnetic(std::size_t axis) const;
  [[nodiscard]] std::complex<double>
  paramagnetic(std::size_t frequency, std::size_t momentum,
               std::size_t left, std::size_t right) const;
};
```

`HoppingResponseAccumulator` is preferred to `CurrentAccumulator`: the raw
event amplitude is a dimensionless signed hopping flux, while the returned
response depends on an explicit Peierls-source convention. The result should
provide all of the following through accumulator/result shape-aware checked
accessors:

- mean density or flux amplitudes, useful as symmetry diagnostics;
- connected density susceptibility;
- the full gauge/flux response tensor;
- the diagonal diamagnetic term;
- the derived paramagnetic current correlation tensor;
- sample count and mode provenance copied into the finished result. The live
  accumulator also exposes its current sample count and the complete free
  `Model` it binds, and run metadata records the `InteractingModel`.

The direct density auto-susceptibility is real and nonnegative configuration by
configuration, so `ContinuousMatsubaraDensityCorrelations` can store `double`
values. Density/flux means are complex. The flux response and derived
paramagnetic response are complex `dimension * dimension` tensors; each mode is
Hermitian (and positive semidefinite for the flux response), while the
diamagnetic vector is real. Mixed-observable results may genuinely be complex
and should have separate types rather than changing the meaning of density auto
correlation.

`paramagnetic(...)` should derive `D-R` from the two authoritative stored terms
rather than own a third parallel tensor that can diverge.

Accumulate each requested `(q,n)` only with its conjugate mode, plus the
physical component tensor. Do not allocate a full `(QF) * (QF)` covariance
matrix: spatial and imaginary-time translation invariance make distinct modes
diagonal in the present model. A future inhomogeneous source would need an
explicit cross-mode request and result shape.

Do not give each result separate public count fields and raw parallel vectors
that can diverge. `MatsubaraModeField<T>` is deliberately the scalar field
owner: it owns one value per mode and is used for the retained transform values
and the continuous density susceptibility. A selected momentum request is not
a complete site lattice, and signed continuous frequencies are not a retained
time axis, so this type should not grow a general trailing-tensor shape.

The semantic result wrappers own and validate their additional storage:

- `ContinuousMatsubaraDensityCorrelations` owns one
  `MatsubaraModeField<double>`, exactly `mode_count()` private complex mean
  amplitudes, its complete free `Model`, and its nonzero sample count.
- `HoppingResponse` is itself the vector/tensor shape owner. For dimension `d`,
  its non-public constructor requires exactly `mode_count()*d` private mean-flux
  values, `mode_count()*d*d` private response values, `d` diamagnetic values,
  one `MatsubaraModeSet`, one complete free `Model`, and a nonzero sample count.
  It checks every product before comparing extents. `paramagnetic` remains
  derived and has no storage buffer.

Only the checked accessors expose those component arrays. Their flat order is
the mode order followed by component, or by left then right component with the
right component varying fastest. This keeps the small reusable owner scalar
without leaving the response tensor shape implicit or publicly mutable.

The retained wrapper becomes a valid-by-construction class rather than the
current public aggregate:

```cpp
class MatsubaraDensityCorrelations {
public:
  MatsubaraDensityCorrelations(
      RetainedGrid source,
      MatsubaraModeField<std::complex<double>> values);

  [[nodiscard]] const RetainedGrid &grid() const noexcept;
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept;
  [[nodiscard]] std::span<const std::complex<double>> values() const noexcept;
  [[nodiscard]] std::complex<double>
  at(std::size_t frequency, std::size_t momentum) const;
};
```

Construction requires the field beta/layout to equal the retained grid,
frequencies to be exactly `0..M-1`, and momenta to be every torus momentum in
flat `TorusLayout` order. It rejects a retained grid whose final DFT row cannot
be represented by `int64_t`. Thus the retained `RetainedGrid` provenance cannot
diverge from the mode field. The continuous result uses the same mode indexing
for a selected subset without making `momentum_points` mean two different
things.

This retained migration intentionally removes the public `frequencies`,
`momentum_points`, and `values` data members. It is a source-breaking pre-1.0
API cleanup: do not retain deprecated parallel fields, because doing so would
preserve the invalid state the owner is meant to remove. The implementation
updates every in-tree caller and test, records the migration in
`cpp/CHANGELOG.md`, and increments the CMake project minor version from `0.5.0`
to `0.6.0`. Callers migrate to `modes().frequency(...)`,
`modes().momentum_count()`, `values()`, and `at(...)`. This completes the
result-shape part of
[repository issue 2](https://github.com/keyehzy/lattice-levy-qmc/issues/2)
without making the continuous implementation depend on later transform
optimisation work in that issue.

Every `observe()` validates exact mode/model compatibility and all input shapes
before changing sums or sample count. It computes per-sample outer products and
candidate updated sums in temporary storage, including all finite and count
checks, then commits with nonthrowing assignments. This avoids a partially
updated accumulator if a late tensor component overflows and gives the same
strong boundary expected of the accepted-state transaction. `finish()` rejects
an empty accumulator.

Construction also rejects a `Model` whose `beta` or `TorusLayout` differs from
the supplied mode set. Projection rejects a context/plan geometry mismatch;
observation then rejects any complete sample-model mismatch, including equal
geometry with a different particle count or hopping value.

Connectedness is an ensemble-statistics policy, not a property of the event
context. Internally, accumulators can share a private complex mode-moment helper
that maintains means and auto/cross co-moments online (the complex Welford
update) without subtracting large final sums. A concrete observable with a
known exact mean, such as density and flux in the present homogeneous model,
should instead accumulate products after analytic centring. A future observable
with an unknown nonzero mean can select estimated centring explicitly and
return the convention and sample count with its result. This keeps the reusable
mechanics general without making every result pretend to have the same
normalisation or contact terms.

A sampling loop remains explicit and small:

```cpp
const Model model = sampler.model().free;
const MatsubaraModeSet modes(
    model.beta(), TorusLayout(model.linear_size(), model.dimension()), request);
const ContinuousMatsubaraPlan plan(modes);
DensityMatsubaraAccumulator density(model, modes);
HoppingResponseAccumulator hopping(model, modes);

for (std::size_t sample = 0; sample < sample_count; ++sample) {
  sampler.sweep(); // Burn-in and thinning remain run-workflow policy.
  const ContinuousMeasurementContext context(sampler.state());
  const ContinuousParticleModes values = continuous_particle_modes(context, plan);
  density.observe(values);
  hopping.observe(values);
}

const auto density_result = density.finish();
const auto hopping_result = hopping.finish();
```

This mirrors the current retained-measurement workflow and does not wait on a
particular future streaming-observer API.

## Density-density estimator

For

\[
\omega_n=\frac{2\pi n}{\beta},\qquad
n_{\mathbf q}(\tau)=\sum_a e^{-i\mathbf q\cdot\mathbf x_a(\tau)},
\]

the per-configuration amplitude is

\[
\rho_{\mathbf qn}=\int_0^\beta
e^{i\omega_n\tau}n_{\mathbf q}(\tau)\,d\tau.
\]

For one particle that is constant on `[a,b)` at physical site `x`, the
contribution is

\[
e^{-i\mathbf q\cdot\mathbf x}\,\Phi_n(a,b),
\]

where

\[
\Phi_n(a,b)=
\begin{cases}
b-a,&n=0,\\[3pt]
e^{i\omega_n(a+b)/2}
(b-a)\,\operatorname{sinc}\!\left(\dfrac{\omega_n(b-a)}{2}\right),
&n\ne0,
\end{cases}
\]

Here `sinc(z) = sin(z)/z`.

The midpoint/sine form avoids subtracting nearly equal complex exponentials.
For very small arguments, implement `sinc` with a series or a tested
near-zero branch. Compute the interval midpoint as `a + (b-a)/2`, not
`(a+b)/2`, so two valid large times cannot overflow during measurement.

With the existing homogeneous fixed-`N` model, the known ensemble mean is

\[
\langle\rho_{\mathbf qn}\rangle=
\beta N\,\delta_{\mathbf q,0}\delta_{n,0}.
\]

Define `delta_rho` by subtracting that known value at amplitude level. The
connected susceptibility matching the existing retained-space convention is

\[
\chi_{nn}(\mathbf q,i\omega_n)
=\frac{1}{\beta V}
  \left\langle |\delta\rho_{\mathbf qn}|^2\right\rangle.
\]

The factor `1/beta` converts the two integrated time arguments in the squared
amplitude into a continuous average over all time origins. Equilibrium time
translation then gives the usual one-time-difference transform

\[
\frac{1}{V}\int_0^\beta d\tau\,
e^{i\omega_n\tau}
\langle\delta n_{\mathbf q}(\tau)
\delta n_{-\mathbf q}(0)\rangle.
\]

Using the analytic mean is both lower variance and safer than subtracting two
large accumulated squares. The accumulator may still report the sampled mean
as a diagnostic. If a future model adds a site-dependent potential or another
source that breaks translation invariance, it must add an explicit estimated-
mean/known-mean policy before calling the result "connected"; it must not
silently reuse the homogeneous centring rule.

This normalisation is not new: it is exactly the continuous-time counterpart
of `retained_grid_matsubara_transform`, whose real-space correlation contains a
translation average divided by `V`.

## Hopping flux, gauge response, and current correlation

Current naming needs an explicit source convention. Put a dimensionless
Peierls field on an oriented positive-axis bond:

\[
H(A)=-t\sum_{\mathbf x,\alpha}
\left[e^{iA_{\mathbf x\alpha}}
a^\dagger_{\mathbf x+\hat\alpha}a_{\mathbf x}
+e^{-iA_{\mathbf x\alpha}}
a^\dagger_{\mathbf x}a_{\mathbf x+\hat\alpha}\right]+H_U.
\]

With \(j^p=-\partial H/\partial A\), the positive-bond paramagnetic current is

\[
j^p_{\mathbf x\alpha}=it\left(
a^\dagger_{\mathbf x+\hat\alpha}a_{\mathbf x}
-a^\dagger_{\mathbf x}a_{\mathbf x+\hat\alpha}\right).
\]

For a hop `e` along axis `alpha` with covering direction `s_e = +/-1`, time
`tau_e`, physical departure `x_e`, and bond midpoint
`b_e = x_e + s_e*e_alpha/2`, define

\[
I_{\alpha}(\mathbf q,i\omega_n)=
\sum_{e:\alpha_e=\alpha}
s_e e^{i\omega_n\tau_e-i\mathbf q\cdot\mathbf b_e}.
\]

The midpoint phase should be evaluated as the departure-site phase times
`exp(-i*s_e*q_alpha/2)`. No half-integer site or ambiguous torus midpoint type
is needed, and a boundary-crossing bond gets the correct phase because allowed
momenta satisfy `exp(-i*q_alpha*L) == 1`.

The full gauge response is

\[
R_{\alpha\beta}(\mathbf q,i\omega_n)=
\frac{1}{\beta V}
\left\langle I_\alpha(\mathbf q,i\omega_n)
I_\beta(\mathbf q,i\omega_n)^*\right\rangle_c.
\]

The equality is most transparent from source differentiation. In the
continuous-time expansion, the Peierls-dependent part of a configuration
weight contains one factor `exp(i*s_e*A(b_e,tau_e))` per hop. Two derivatives
give

\[
-\frac{\delta^2\log Z}{\delta A\,\delta A}
=\langle I I\rangle_c,
\]

and hence the curvature of \(F=-\log Z/\beta\) is the positive event-flux
covariance divided by `beta`.
Taking the same derivatives in the Hamiltonian representation produces the
diamagnetic term minus the time-ordered paramagnetic current correlator. This
is also why a future event observable must document its source derivative and
contact terms; an impulse covariance is not automatically the correlation of
an operator with the same informal name.

For the source-free, time-reversal-invariant model in this repository,
the exact mean flux is zero. As with density, report its sampled mean as a
diagnostic and require a new explicit centring policy if the model later gains
a background gauge field.

Let `K_alpha` be the number of hopping events along axis `alpha`. The
diamagnetic estimator and paramagnetic operator correlation in the convention
above are

\[
D_\alpha=\frac{\langle K_\alpha\rangle}{\beta V}
=-\frac{\langle H_{\mathrm{kin},\alpha}\rangle}{V},
\]

and therefore `sum_alpha D_alpha == -<H_kin>/V` with the existing total
event-count estimator.

\[
\Lambda^p_{\alpha\beta}
=\delta_{\alpha\beta}D_\alpha-R_{\alpha\beta}.
\]

Thus an event-flux correlation is **not** the paramagnetic current-current
correlator by itself. It is the full Peierls/gauge response, including the
same-event contact contribution. Returning `flux_response`, `diamagnetic`, and
`paramagnetic` together prevents a caller from losing this distinction. The
older worldline literature also distinguishes physical paramagnetic current
from the path "pseudocurrent"/flux estimator; see Batrouni et al.,
[Universal Conductivity in the Two dimensional Boson Hubbard Model](https://arxiv.org/abs/cond-mat/9302037).

The library should not initially return conductivity. Charge, `hbar`, lattice
spacing, analytic continuation, order of limits, and possible subtraction of a
zero-frequency stiffness contribution are user-level physics choices. The
measurement layer returns imaginary-time/Matsubara response data with its
source convention. In particular, the zero-momentum and zero-frequency limits
distinguish stiffness and transport diagnostics; see Scalapino, White, and
Zhang,
[Insulator, metal, or superconductor: The criteria](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.7995).

In the repository's `k_B = 1`, unit-lattice-spacing convention,
`rho` has units of inverse energy, `chi_nn` has units of inverse energy per
site, `I` is dimensionless, and `R`, `D`, and `Lambda^p` have units of energy
per site. No electric charge factor is included.

## Exact cross-checks from the shared phase convention

The following identities are strong enough to define the convention in tests.

### Canonical density identities

\[
\rho_{\mathbf0,0}=\beta N,\qquad
\rho_{\mathbf0,n}=0\quad(n\ne0),
\]

so every connected `q == 0` density susceptibility is exactly zero in the
fixed-`N` ensemble. A configuration with no events has zero
nonzero-frequency density amplitude for every momentum.

### Winding identity

At zero momentum and frequency,

\[
I_\alpha(\mathbf0,0)=\sum_{e:\alpha_e=\alpha}s_e=L W_\alpha.
\]

Consequently the zero-mode response connects directly to the existing winding
and twist-curvature estimators. In the conventions of
[`MEASUREMENTS.md`](MEASUREMENTS.md),

\[
R_{\alpha\alpha}(\mathbf0,0)
=\frac{L^2}{V}\frac{\langle W_\alpha^2\rangle}{\beta}
=\frac{L^2}{V}
\left.\frac{\partial^2F}{\partial\phi_\alpha^2}\right|_{\phi=0}.
\]

### Lattice continuity/Ward identity

A hop changes the density mode by

\[
e^{-i\mathbf q\cdot(\mathbf x+s\hat\alpha)}
-e^{-i\mathbf q\cdot\mathbf x}
=-2is\sin(q_\alpha/2)e^{-i\mathbf q\cdot\mathbf b}.
\]

Integration by parts around the closed imaginary-time circle therefore gives,
for every configuration and every requested signed Matsubara index,

\[
\omega_n\rho_{\mathbf qn}
=2\sum_\alpha\sin(q_\alpha/2)
 I_\alpha(\mathbf q,i\omega_n).
\]

This is the most valuable deterministic test of event ordering, bond midpoint,
spatial sign, and time-transform sign at once.

### Conjugation and time-origin rotation

Density is a site field, so with
`bar(k)_alpha = (L-k_alpha) % L`,

\[
\rho(\mathbf k,i\omega_n)^*=\rho(\bar{\mathbf k},-i\omega_n).
\]

Flux is instead evaluated at oriented bond midpoints. With momentum components
stored canonically in `[0,L)`, replacing the signed wavevector `-q` by its
canonical representative adds a reciprocal vector. Integer site coordinates
are periodic under that shift, but the half-coordinate on the flux component's
bond axis contributes a gauge sign:

\[
I_\alpha(\mathbf k,i\omega_n)^*
=\eta_\alpha(\mathbf k)
 I_\alpha(\bar{\mathbf k},-i\omega_n),\qquad
\eta_\alpha(\mathbf k)=
\begin{cases}
+1,&k_\alpha=0,\\
-1,&k_\alpha\ne0.
\end{cases}
\]

Equivalently, flux obeys the plain real-field conjugation identity when `-q`
is kept as an unwrapped signed wavevector rather than canonicalized. This
bond-basis gauge is also what keeps the Ward factor in the stated
`2*sin(q_alpha/2)` form for canonical `q_alpha`.

For the repository's `rotate_configuration_time_origin(state, shift)`
convention, a cyclic time-origin rotation gives
`X' = exp(-i*omega_n*shift) X` (with seam events handled by the same event
sweep), leaving every auto response invariant. These should be exact/tight
deterministic tests, not only Monte Carlo comparisons.

## Extension to other observables

The segment-plus-impulse formula covers a useful, well-defined family.

| Observable | Segment contribution | Event contribution | Extra state |
| --- | --- | --- | --- |
| Density | `sum_a exp(-i*q*x_a)` | none; hop updates the cached value | Particle positions |
| On-site pair density | `sum_x choose(n_x,2)` or its spatial modes | none | Sparse occupancies |
| Occupation projector/moment | Function of `n_x` | none | Sparse occupancies |
| Potential/diagonal energy density | Function of the current occupation state | none | Sparse occupancies |
| Signed particle flux | none | Direction times bond phase | Hop departure/arrival |
| Hopping activity | none | Unsigned event/bond weight | Hop departure/arrival |
| Density-flux response | Density residence amplitude | Flux impulse amplitude | Cross-mode moment accumulator |

Adding one of these should require an observable-specific projector and result
type, while reusing the context, exact interval kernel, plan, shape owner, and
mode-moment accumulation. Once at least three projectors demonstrate a stable
common callable contract, a constrained internal `project_continuous_signal`
template may remove boilerplate. It should remain private until there is a real
external custom-observable requirement.

This abstraction does **not** make every quantum observable measurable in the
closed worldline sector. Single-particle Green functions and other off-diagonal
operator insertions generally require an open-worldline/worm sector or a
separately derived improved estimator. They should not be represented as empty
segment/event callbacks. The original continuous-time worldline construction
also treats worldline discontinuities as the route to Green functions; see
Prokof'ev, Svistunov, and Tupitsyn,
[Exact, Complete, and Universal Continuous-Time Worldline Monte Carlo](https://arxiv.org/abs/cond-mat/9703200).

For an imaginary-time result `C(r,tau)` rather than Matsubara modes, add a
separate requested-lag backend later. It can compute exact overlap lengths of
piecewise-constant periodic intervals at each requested lag. Such lags are
evaluation points, not a Trotter or retained measurement grid. Inverting a
finite Matsubara set is useful for presentation but is not an exact continuous-
time `tau` estimator and may ring at the frequency cutoff.

## Numerical and complexity details

- Use `double` times and `std::complex<double>` amplitudes.
- Compute frequencies from signed integer indices only after checking that the
  conversion and `2*pi*n/beta` are finite; a nonzero index that underflows to
  zero is also rejected.
- `ContinuousMatsubaraPlan` supports
  `|n| <= kMaximumAbsoluteFrequencyIndex == 1'048'576`. It validates magnitude
  through an unsigned representation, without first negating the signed value,
  and therefore rejects `int64_t` minimum safely. Supporting larger indices is
  a future numerical-backend change, not an undocumented relaxation.
- Form event phases from dimensionless cycles. For an interior time, compute
  `cycles = double(n) * (tau / beta)`, reduce it with
  `std::remainder(cycles, 1.0)`, and only then multiply by `2*pi`. For the sine
  in an interval `sinc`, reduce `double(n) * ((b-a) / beta)` modulo `2` before
  evaluating the numerator, while retaining the unreduced value in the
  denominator. Return a zero numerator directly when that reduced cycle value
  is exactly `0` or `+/-1`, rather than evaluating `sin(0)` or `sin(+/-pi)`.
  This avoids asking the standard library to reduce a large angle and avoids
  multiplying a separately rounded physical frequency by `tau`.
- At the maximum supported index, deterministic unit phases at analytically
  known rational normalized times must agree within
  `64*epsilon*(1+abs(n))` in absolute complex norm. This is the binary64 phase
  contract; it does not promise meaningful modes whose oscillation is below the
  resolution of the stored event times.
- Return the time phase exactly as `1 + 0i` at normalised times `0` and `beta`
  for every integer Matsubara index, preserving seam periodicity explicitly.
- Use the midpoint/sine interval kernel above; special-case `n == 0` exactly.
- Return `Phi_n(0,beta) == 0` exactly for `n != 0`, rather than evaluating
  `sin(pi*n)` in floating point.
- Form site phases after reducing each integer product `k_alpha*x_alpha`
  modulo `L` with an overflow-safe modular product. Converting an unreduced
  product to `double` needlessly loses the exact torus periodicity on large
  one-dimensional layouts.
- Fill `q == 0` density amplitudes from the canonical identities instead of
  summing many intervals, so exact conservation is not degraded by roundoff.
- Accumulate the signed displacement per axis with checked integer arithmetic
  and use it for `I(0,0)`, preserving the winding identity before conversion to
  `double`.
- Use a tested compensated real/imaginary sum for long, cancellation-heavy
  density and flux mode sequences unless benchmarks show an unacceptable cost;
  at minimum compare it with the direct sum over large-event fixtures before
  fixing the numerical contract.
- Apply a hop to physical sites through `TorusLayout::shifted`, while retaining
  its covering direction for winding and current orientation.
- Calculate checked shapes for `N_q*N_omega`,
  `N_q*N_omega*dimension`, and response tensors before allocation.
- Reject nonfinite projected amplitudes or accumulated response entries with a
  descriptive overflow error before committing an observation.
- Form response diagonals with `std::norm` and mirror one computed tensor
  triangle by conjugation, so Hermiticity is structural rather than a
  post-processing tolerance.
- Do not force a full-volume momentum set. Selected momenta and frequencies are
  important when `V` or the desired high-frequency cutoff is large.
- Keep direct sums as the correctness baseline. Dense-frequency recurrences,
  site-phase tables, k-way event merging, SIMD, or FFTs require representative
  benchmarks and exact equivalence tests.

With `E` hops, `Q` requested momenta, and `F` requested frequencies, context
construction is initially `O(N + E log E)` time and `O(N + E)` storage. A
straight combined projector is `O(NQ + EQF)` time and
`O(QF(1+d) + d)` per-sample output storage. This is independent of an arbitrary
imaginary-time resolution. For density alone a path-local `O(N+E)` traversal
without global sorting is possible (the `QF` projection work remains), but one
shared ordered context is the more valuable baseline once occupancy observables
are included; benchmarks can justify a density-only fast path later.

## Files and dependency boundary

A coherent implementation would use:

```text
cpp/include/qmc/matsubara_modes.hpp
    shared geometry-only mode descriptor and checked mode-field template

cpp/src/matsubara_modes.cpp
    non-template mode validation, indexing, and accessors

cpp/include/qmc/continuous_observables.hpp
    public continuous plan, context, primitive mode sample, results/accumulators

cpp/src/continuous_event_sweep.hpp
cpp/src/continuous_event_sweep.cpp
    private event collection/grouping/replay used first by continuous
    measurements; interaction.cpp migrates to it in implementation step 6

cpp/src/continuous_observables.cpp
    exact interval projection and accumulators

cpp/tests/test_continuous_observables.cpp
    deterministic boundary, identity, result-shape, and ED regressions

cpp/tests/test_interacting_statistical.cpp
    sampled finite-U and high-frequency convergence checks

cpp/CMakeLists.txt
cpp/tests/CMakeLists.txt
    source and test registration in qmc_ideal/qmc_interacting as assigned above
```

`qmc/ideal.hpp` should include `qmc/matsubara_modes.hpp`, and
`qmc/interacting.hpp` should include the new continuous public header. The
geometry-only mode descriptor belongs with the lattice-transform code in
`qmc_ideal`; all sources that mention continuous paths belong to
`qmc_interacting`. `qmc_ideal` must not depend on continuous paths. Reusable
spatial phase planning may later move to an ideal-independent internal utility
when retained transforms adopt the same convention.

Do not move the accepted-state `OccupancyIndex` into the public measurement
layer. It is mutable transactional sampler state optimised for replacements.
The measurement sweep needs immutable per-configuration replay and can share
small event/state primitives without sharing that ownership model.

Because this adds public result and measurement APIs, implementation should
also update `cpp/CHANGELOG.md`, the continuous-observable section of the user
documentation, and the example output schema if the interacting demo starts
emitting these results. The retained-wrapper migration unconditionally updates
the ideal demo, its output verification, and retained-observable documentation.
Keep the derivation here rather than duplicating it in source comments; public
declarations need only concise units, signs, indexing, ownership, and failure
behavior.

## Verification plan

### Deterministic unit tests

- Context construction preserves `configuration.event_count()`, every stored
  departure/arrival/direction is locally consistent, and replay closes onto the
  permutation-successor seam positions.
- Static paths: exact interval amplitudes, including small `omega*(b-a)`.
- `q == 0` canonical density identities for zero, positive, and negative
  frequency indices.
- `I(0,0) == L*W` in every axis, including boundary-crossing hops.
- The configuration-level Ward identity over several dimensions, momenta, and
  signed frequencies.
- Conjugation under `(q,n) -> (-q,-n)`.
- A global covering-space translation by `r` multiplies both primitive fields
  by `exp(-i*q*r)` and leaves auto responses unchanged.
- The known phase under `rotate_configuration_time_origin`; auto correlations
  remain unchanged.
- Equal-time event groups, multiple events on one particle, events at `0` and
  `beta`, and right-continuous seam handling.
- Reuse the coincident `0`, internal-seam, and `beta` fixture in
  `CursorRotationMatchesCoincidentSeamTraversalExactly` so rotation and
  measurement prove the same boundary convention rather than maintaining two
  almost-identical handcrafted cases.
- Empty-particle, no-event, `L == 1`, `L == 2`, and multiple-cycle
  configurations. In particular, preserve the covering direction when two
  directions reduce to the same physical arrival site.
- Mode/plan validation: malformed momentum dimension/component, duplicates,
  unrepresentable frequency, checked-extent overflow, acceptance at both signs
  of `kMaximumAbsoluteFrequencyIndex`, and rejection immediately beyond it and
  at `int64_t` minimum; projection rejects a context/plan geometry mismatch and
  accumulators reject a complete model mismatch.
- Maximum-index time phases at rational normalized times satisfy the stated
  binary64 error bound, while phases at `0` and `beta` and full-period interval
  transforms retain their exact special-case values.
- Extreme one-dimensional layouts preserve torus-periodic site phases without
  overflowing `k*x` or relying on large floating-point angle reduction.
- For representative full momentum sets, plan site phases match the existing
  `phase_for_indices`/retained-transform sign and ordering to a tight numerical
  tolerance; the extreme-layout case uses an integer modular reference.
- Accumulator validation happens before mutation; empty `finish()` fails and a
  single compatible observation preserves the per-configuration result. Every
  finished result reports the accumulator's exact nonzero sample count.
- The retained Matsubara wrapper rejects a field with a different beta, layout,
  frequency sequence, momentum sequence, or value extent; migrated transform
  values and ordering match the pre-migration implementation.
- `HoppingResponse` construction rejects each wrong or overflowing vector/tensor
  extent before publishing a result.
- Result accessors reject out-of-range frequency, momentum, and tensor axes.

### Physics comparisons

- Small canonical exact-diagonalisation/Lehmann comparisons of
  `chi_nn(q,i*omega_n)`.
- Compare event response against `D - Lambda^p` from exact diagonalisation at
  both zero and finite momentum. Comparing event flux directly with
  `Lambda^p` would test the wrong quantity.
- Check `D_alpha` against the axis-resolved event-count kinetic estimator.
- Verify the high-frequency approach `R_aa -> D_a` and
  `Lambda^p_aa -> 0` statistically.
- At `U == 0`, compare the exact continuous result with the deterministic
  canonical/Lehmann answer; for finite `U`, use the existing small-system ED
  machinery with controlled Markov-chain errors.
- As a secondary regression, show convergence of the retained-grid density
  transform toward the continuous result as `M` increases. The retained value
  is not the reference at finite `M`.

Hard-coded small Lehmann tables and algebraic identities belong in the fast
interacting unit target. Monte Carlo convergence, high-frequency limits, and
finite-`U` sampled comparisons belong in the labelled statistical target with
documented seeds, sample counts, and uncertainty-derived tolerances.

Autocorrelation analysis, blocking/jackknife errors, covariance between output
modes, and analytic continuation are run-analysis concerns. The core
accumulator should retain raw sample count and well-defined first/second mode
moments so those tools can be built without changing the physics convention.

### Design-time numerical checks

Several independent scratch checks were used to challenge the formulas while
writing this design; they are evidence for the convention, not substitutes for
committed C++ tests.

- Exact amplitudes from 20 sampled `N=5`, `L=6`, `d=2` ideal continuous
  configurations satisfied the Ward identity for several zero/nonzero
  momenta and positive/negative frequencies with maximum residual below
  `2e-11`.
- A one-particle Lehmann calculation at `N=1`, `L=3`, `beta=0.8`, and `t=1`
  gave `D=0.5130943014`; 80,000 independently sampled ideal paths gave the
  event-count estimate `0.5144218750`. At `q=2*pi/3`, the exact density values
  for `n=0,1` were `0.1915077492` and `0.0217766472`, versus event-integral
  estimates `0.1913205998` and `0.0218628946`.
- The same Lehmann source-derivative calculation confirms the contact-term
  interpretation. At finite `q=2*pi/3`, the exact full responses for `n=0,1`
  were `0` and `0.4477643599`; 100,000 independent paths gave approximately
  `1e-32` and `0.4462462370`. The corresponding physical comparison is
  `R=D-Lambda^p`, not `R=Lambda^p`.
- A finite-interaction importance-reweighting check used 40,000 independent
  ideal paths at `N=2`, `L=3`, `beta=0.8`, `t=1`, and `U=1.2`. At
  `q=2*pi/3`, the reweighted density values for `n=0,1` were approximately
  `0.32716, 0.04820` versus Lehmann values `0.32798, 0.04773`; the `n=1`
  response was `0.99112` versus `0.98151`, and the diamagnetic estimate was
  `1.16611` versus `1.16136`.

The statistical differences are consistent with the deliberately simple
unblocked Monte Carlo diagnostics. A production test should use deterministic
Lehmann tables where possible and an explicitly budgeted statistical tolerance
where sampling is essential.

## Rejected alternatives

### Repeated `positions_at(tau)` on a measurement grid

This introduces an arbitrary integration error, repeatedly scans paths, makes
cost depend on a nonphysical resolution, and blurs the documented distinction
between retained ideal paths and exact interacting paths.

### Put dynamical arrays in `InteractingObservables` or `run()`

The scalar bundle is cheap per sample and currently materialised by `run()`.
Large mode tensors have different lifetime, storage, and streaming needs.
Binding them to the sampler would couple Markov transitions, statistical
accumulation, and output policy.

### Let every accumulator rescan `ContinuousConfiguration`

Density, flux, pair density, and later observables would each reproduce seam
normalisation, site reduction, event ordering, and phase conventions. A
per-configuration context is the existing codebase's answer to this repeated
derived-state problem.

### Public virtual/callback `Observable`

This would introduce lifetime and exception contracts, per-event dynamic
dispatch, and an abstract framework before the supported estimator families are
known. Concrete projectors over one context and plan provide the useful reuse
with less API commitment.

### Call the raw event correlation "current-current"

It hides the diamagnetic/contact term and conflicts with the operator meaning
of the paramagnetic current correlator. The API should make the source
derivative and returned decomposition explicit.

## Implementation order

1. **Completed 2026-07-22:** Add `MatsubaraModeSet`/`MatsubaraModeField`,
   migrate the retained result wrapper with exact ordering/value equivalence,
   and add the continuous plan.
2. **Completed 2026-07-22:** Add the owning continuous measurement context and
   deterministic event-sweep boundary tests.
3. **Completed 2026-07-22:** Add density/flux primitive modes together and make
   the Ward identity the central convention test.
4. Add the density accumulator and small-system Lehmann comparisons.
5. Add the hopping response accumulator returning `R`, `D`, and `Lambda^p`,
   followed by zero/finite-momentum ED comparisons.
6. Migrate full pair-overlap evaluation to the shared private event sweep only
   after exact equivalence tests prove unchanged action and tie semantics.
7. Add an occupancy-based diagonal projector as the third use case; then decide
   whether a private generic projection template is clearer than three small
   concrete loops.
8. Add requested-lag imaginary-time output, block/covariance tooling, or
   transform optimisations only in response to a concrete workflow and
   benchmark.

## Design conclusion

The stable public concepts should be **configuration event geometry**,
**validated Matsubara modes**, **per-configuration primitive amplitudes**, and
**typed ensemble responses**. The reusable mathematical mechanism is a sum of
exact segment integrals and event impulses. Keeping those layers separate
captures density, hopping flux, diagonal occupation observables, and mixed
responses while preserving this repository's preference for immutable
provenance, value ownership, checked shapes, explicit conventions, and small
composed types.
