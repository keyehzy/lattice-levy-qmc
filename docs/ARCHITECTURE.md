# Architecture and algorithms

This document describes the current implementation as an executable model,
not merely as a list of Python functions. It records the probability laws,
data flow, invariants, update semantics, and costs that explain why the code
works.

## 1. Conceptual layers

The project is organized as four logical layers even though it currently uses
only two core source files:

```text
one-particle numerics
  Bessel count law, lattice bridges, torus trace, winding sectors
                |
                v
canonical ideal gas
  partition recursion -> cycle labels -> ideal path configuration
                |
                v
continuous configuration geometry
  event paths, permutation endpoints, torus occupancy, winding
                |
                v
finite-U Markov chain
  free proposals -> exact interaction action -> Metropolis -> observables
```

`python/lattice_levy.py` implements the first two layers and a grid-observed ideal
configuration. `python/interacting_lattice_levy.py` reuses the ideal thermodynamics
and winding machinery, introduces sparse continuous-time paths, and implements
the final two layers.

## 2. Mathematical conventions

The free Hamiltonian is

\[
H_0=-t\sum_{\langle ij\rangle}(|i\rangle\langle j|+|j\rangle\langle i|).
\]

On the infinite one-dimensional covering lattice,

\[
K_{\mathbb Z}(a,b;\tau)
=\langle b|e^{-\tau H_0}|a\rangle
=I_{|b-a|}(2t\tau).
\]

Coordinates factorize in \(d\) dimensions. A physical site is a vector in
\(\mathbb Z_L^d\), while sampling occurs in the covering space
\(\mathbb Z^d\). A closed torus path may therefore end at `start + L*w`, where
`w` is its integer winding vector.

The interacting target measure is written relative to the normalized ideal
canonical measure \(\pi_0\):

\[
d\pi_U(R)\propto d\pi_0(R)e^{-S_U(R)},\qquad
S_U(R)=U A(R),
\]

where

\[
A(R)=\int_0^\beta d\tau\sum_x\binom{n_x(\tau)}{2}.
\]

Every proposal in the interacting module is sampled from the appropriate
conditional or marginal ideal measure. Its free-path factors cancel from the
Metropolis-Hastings ratio, leaving only `exp(-delta_action)`.

## 3. Free bridge sampling

### 3.1 Conditioned jump counts

A continuous-time symmetric random walk in one coordinate has independent
right- and left-jump counts

\[
N_+,N_-\sim\operatorname{Poisson}(\lambda),\qquad \lambda=t\tau.
\]

Condition on the endpoint displacement
\(N_+-N_-=\Delta=b-a\). If \(D=|\Delta|\) and `m` is the smaller count, then

\[
P(m\mid D)\propto
\frac{\lambda^{2m+D}}{m!(m+D)!},\qquad m\ge 0.
\]

`_sample_bessel_pair_count` evaluates this distribution in log space. It
starts beyond an estimate of the mode, increases the retained support until a
monotone-ratio geometric bound proves the omitted mass is below `tail_tol`,
then samples by CDF inversion. This helper is private in Python but is shared
by both core modules.

Once `m` is known, the sign of \(\Delta\) determines which jump count is
`m + D` and which is `m`.

### 3.2 Grid-observed midpoint and bridge

For a bridge split into left and right durations, each of the conditioned
right and left jumps independently falls in the left interval with probability
`tau_left / total`. Two binomial draws therefore produce the midpoint without
enumerating candidate sites. The resulting law is

\[
P(k\mid a,b)=
\frac{I_{|k-a|}(2t\tau_L)I_{|b-k|}(2t\tau_R)}
     {I_{|b-a|}(2t(\tau_L+\tau_R))}.
\]

`sample_bridge_covering` applies this midpoint sampler recursively. It stores
both endpoints, uses an explicit stack, and splits an index interval at
`width // 2`. Unequal physical durations on the two sides make arbitrary
`n_steps` valid; there is no power-of-two restriction.

### 3.3 Continuous bridge

`sample_continuous_bridge_covering` samples the same conditioned counts in
each coordinate. Given those counts, it draws independent uniform event times
inside `(0, duration)`, creates a signed unit jump for each event, and performs
a stable time sort. The result is a `ContinuousPath`; no observation grid is
introduced.

`split_continuous_path` partitions a long event path into local-time pieces.
`resample_path_interval` replaces `(tau0, tau1]` with a fresh fixed-endpoint
free bridge while retaining events at or before `tau0` and after `tau1`.
These endpoint choices agree with the right-continuous position convention.

## 4. Finite-volume ideal gas

### 4.1 Torus trace

For a cycle lasting `s`, the exact one-particle trace is evaluated by a finite
momentum sum:

\[
Z_1(s)=\left[\sum_{n=0}^{L-1}
e^{2ts\cos(2\pi n/L)}\right]^d.
\]

`log_one_particle_trace_torus` uses `logsumexp`, so it remains stable when the
unscaled exponentials would overflow. `periodic_kernel_scaled_1d` provides the
related point-to-point torus kernel for direct torus midpoint sampling; the
main configuration samplers instead work in covering space so that winding is
retained explicitly.

### 4.2 Canonical partition recursion

Let `z[ell] = Z_1(ell*beta)`. The ideal canonical partition function satisfies

\[
Z_0=1,\qquad
Z_n=\frac{1}{n}\sum_{\ell=1}^{n}z_\ell Z_{n-\ell}.
\]

`CanonicalEnsemble` computes `log_z[1:N+1]` and `log_Z[0:N+1]` entirely in log
space and keeps them bound to the validated model that produced them. Its
`sample_cycles` operation then repeatedly samples the length of the cycle
containing the smallest remaining label with probability

\[
P(\ell\mid n)=\frac{z_\ell Z_{n-\ell}}{n Z_n}.
\]

The other labels in that cycle are selected uniformly without replacement and
shuffled to choose the directed cyclic order. Removing the chosen labels and
repeating produces a labeled permutation with its exact canonical weight.

### 4.3 Winding

For a closed one-dimensional ring path of duration `s`,

\[
P(w)\propto I_{|w|L}(2ts),\qquad w\in\mathbb Z.
\]

`sample_winding_1d` evaluates exponentially scaled Bessel values (`ive`),
expands a symmetric support until a geometric bound controls both omitted
tails, and samples a signed winding. Dimensions are independent under the free
Hamiltonian.

### 4.4 Configuration assembly

For each sampled cycle of length `ell`, both ideal constructors:

1. draw a base site uniformly in `[0, L)^d`;
2. draw a winding vector for duration `ell*beta`;
3. draw one covering-space bridge from `base` to `base + L*winding`;
4. split the long loop into `ell` consecutive pieces of duration `beta`;
5. assign those pieces to the ordered particle labels and record the
   permutation successor of every label.

`sample_ideal_boson_configuration` retains `ell*M + 1` equally spaced points
on the long loop. `sample_ideal_continuous_configuration` retains its jump
events and splits them at multiples of `beta`.

## 5. Data model and invariants

| Type | Important fields | Invariants and ownership |
| --- | --- | --- |
| `CyclePath` | labels, base point, winding, covering and torus paths | Frozen ideal-skeleton record for one long cycle; displacement is `L*winding`; torus endpoints coincide |
| `IdealBosonConfiguration` | model parameters, tuple of cycles, permutation, dense world-line arrays, `log_ZN` | Frozen top-level ideal sample; dense arrays have shape `(N, M+1, d)` |
| `ContinuousPath` | duration, start/end vectors, sorted times, jump matrix | Frozen, value-like record; each jump has one `+1` or `-1`; sum of jumps exactly connects start to end; positions are right-continuous |
| `ContinuousConfiguration` | model parameters, cycle list, permutation, list of paths, `log_Z0_N` | Mutable path container used by moves; cycles partition labels; path `i` ends at the start of `permutation[i]` modulo `L` |
| `MoveStatistics` | attempts, accepts, topology changes | Mutable counters; rates are undefined before an attempt |
| `InteractingLatticeLevySampler` / `InteractingSampler` | model, RNG, state, cached overlap/action, occupancy index, counters | Owns the Markov chain; `action == U*pair_overlap`; cached values describe the accepted state |

Arrays use signed 64-bit integers for positions and jumps and double precision
for times and log weights. The frozen dataclasses are frozen at the attribute
level, but their NumPy arrays are not marked read-only. The sampler treats
paths as immutable and only replaces whole path references; external callers
should do the same. `ContinuousConfiguration.copy()` copies container state
but shares the existing `ContinuousPath` objects under that convention.

`ContinuousConfiguration.validate()` is the authoritative structural check.
It verifies permutation and cycle consistency, dimensions and durations, and
endpoint matching modulo `L`. `total_winding()` additionally asserts that the
sum of covering displacements is divisible by `L` in every dimension.

## 6. Exact interaction action

`pair_overlap_time` performs a sweep over all hopping events:

1. reduce every starting position modulo `L` and build an occupancy map keyed
   by the `d`-component site tuple;
2. compute the initial number of on-site pairs, `sum n*(n-1)/2`;
3. collect `(time, particle, jump)` from all paths and sort by time;
4. integrate the current pair count over the interval before each event time;
5. update the departure and arrival occupancies in constant expected time;
6. process equal-time events as one zero-duration group, then integrate the
   final interval to `beta`.

This is exact for the piecewise-constant paths. It costs `O(E log E + N)` time
and `O(E + N)` auxiliary memory for `E` events and remains the authoritative
full validator.

Accepted-state updates use a sparse site-by-site occupancy ledger. It
integrates the current occupancy along removed and added paths, so each affected
pair is counted exactly once; rejected proposals roll the ledger back.

## 7. Markov moves and detailed balance

The sampler starts from an independent ideal configuration and caches its pair
overlap and action. There are five move families:

| Move | Proposed change | Preserved | Changes cycle topology? |
| --- | --- | --- | --- |
| Segment | One fixed-endpoint interval on one labeled path | Path start/end, permutation, winding | No |
| Cycle | All paths belonging to one selected cycle | That cycle's ordered labels and length | No; base point, events, and winding may change |
| Stitch | `k` exact torus bridges in a closed slab with a permanent-sampled suffix matching | Exterior physical occupancy | Yes; rearranges selected successors |
| Time shift | Uniform cyclic rotation of every closed loop in imaginary time | All physical observables and topology | No |
| Global | A complete independent ideal configuration | Only model parameters | Yes |

Segment and cycle proposals use exact free conditionals. Stitching additionally
samples a permanent-normalized endpoint matching with products of exact torus
kernels, then samples the exact continuous-time bridges. The free factors
therefore cancel and the Metropolis decision again contains only the action
difference. Time shifts are measure-preserving and rejection-free. The global
move replaces the whole configuration only on acceptance. Interaction-corrected
moves use

```text
accept immediately if delta_action <= 0
otherwise accept if log(U[0,1)) < -delta_action
```

and explicitly reject exponents beyond the useful double-precision range.
At `U=0`, every nonempty proposed move is accepted. Pair stitches alone
generate all permutation sectors; optional `k<=8` stitches rearrange several
successors at once while remaining entirely in the space of closed
configurations. A small uniform-partner component prevents the locality
heuristic from disconnecting the chain.

`random_seam_stitch_sweep(m)` applies a palindromic kernel `A B**m A`, where
`A` is the reversible uniform time-origin rotation and `B` is the fixed-seam
random-scan stitch kernel. This macro-kernel satisfies detailed balance and
amortizes one spatial bucket build over all stitch attempts. See
`RANDOM_SEAM_STITCH.md` for the derivation. `sweep()` remains a configurable
low-level scheduler.

## 8. Observables

For total jump count `K` and pair-overlap time `A`, the implemented estimators
are

\[
E_{\mathrm{kin}}=-K/\beta,\qquad
E_{\mathrm{int}}=UA/\beta,\qquad
E=E_{\mathrm{kin}}+E_{\mathrm{int}}.
\]

The double occupancy per site is `A/(beta*L**d)`. Total winding is the sum of
all covering-space path displacements divided by `L`; the telescoping
permutation endpoints make this an integer vector. Cycle lengths are reported
directly from the current topology.

The estimator functions and `observables()` expose the same formulas. The
sampler method uses its cached overlap to avoid one redundant action sweep.

## 9. Public interfaces

`lattice_levy.__all__` exposes:

- the two ideal data records;
- covering-space midpoint and grid bridge samplers;
- the scaled periodic kernel and direct torus midpoint sampler;
- torus trace and canonical partition recursion;
- cycle-label and winding samplers;
- the complete ideal skeleton constructor;
- a finite-window midpoint PMF evaluator used for verification.

`interacting_lattice_levy.__all__` exposes:

- continuous path and configuration records;
- the finite-`U` sampler and move statistics;
- continuous bridge, split, interval-resampling, and ideal configuration
  constructors;
- exact overlap/action functions and energy/double-occupancy estimators.

Names beginning with `_` are implementation details. The notable coupling is
that the interacting module imports `_sample_bessel_pair_count` from the ideal
module. A port should move that primitive into a shared numerical/free-path
component rather than reproduce the Python module dependency.

## 10. Cost summary

Let `V=L**d`, `E` be the number of continuous jump events, and `W` be the
retained winding support.

| Operation | Time | Storage |
| --- | --- | --- |
| Canonical ensemble construction | `O(N*L + N**2)` | `O(N)` |
| Cycle-label draw | `O(N**2)` in the list-based implementation | `O(N)` |
| Winding draw per coordinate | `O(W)` Bessel evaluations per support trial | `O(W)` |
| Ideal skeleton | `O(N*M*d)` midpoint draws plus inversion work | `O(N*M*d)` |
| Continuous bridge | Jump-count inversion plus `O(E log E)` sort | `O(E*d)` in the current jump matrix |
| Pair-overlap action | `O(E log E + N)` | `O(E + N)` |
| Fixed-size local action update | Expected local event/timeline work at fixed density | Sparse per-site timelines |
| Random-seam stitch macro-step | `O(N)` bucket build plus `O(k**2 + k*2**k)` matching work per attempt, `k<=8` | `O(E + N + 2**k)` |
| Global finite-`U` proposal | Ideal sample plus one full action evaluation | Temporary full configuration |

The torus trace uses `L` momentum terms and then multiplies its logarithm by
`d`; it does not enumerate `V` sites.

## 11. Verification evidence

The tests are organized around the derivation:

- Bessel convolution checks the bridge normalization identity.
- Empirical midpoint samples are compared with the exact PMF, including a
  widely separated endpoint case that defeats fixed spatial windows.
- The torus trace is compared with a direct momentum sum.
- The canonical recursion is compared with explicit permutation enumeration.
- Non-power-of-two bridge grids and full ideal configuration invariants are
  checked directly.
- Continuous bridges, splitting, and interval replacement are checked for
  endpoint and event preservation.
- Open torus bridges are checked for the requested endpoint modulo `L`.
- The overlap integral is checked on a hand-built piecewise path.
- Stitch updates are required to change topology, preserve structural
  invariants, and keep their incremental action equal to a full overlap sweep.
- Time-origin rotations are checked to preserve event count, winding, and
  interaction action.
- All move families are required to accept at `U=0`, and finite-`U` sweeps are
  repeatedly validated for structural and cache consistency.
- `python/validate_interacting_ed.py` supplies an independent small-system physics
  comparison using exact diagonalization.

These checks cover correctness of the main formulas and invariants. They do
not constitute production convergence diagnostics for arbitrary simulation
parameters.
