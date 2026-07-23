# Ideal canonical measurements

The C++ ideal-gas library exposes deterministic canonical measurements and
configuration estimators through `qmc/observables.hpp`. Deterministic results
use the same finite-torus momentum sums and log-domain canonical table as the
sampler. Configuration estimators act on the exact dense skeleton at its `M`
distinct retained times. The endpoint at `beta` is excluded because it is the
permuted copy of time zero.

Throughout this document, `k_B = 1`, momenta are
`q_alpha = 2*pi*k_alpha/L`, and flat lattice indices use axis zero as the
least-significant base-`L` digit.

## Deterministic canonical observables

`canonical_thermodynamics` returns `F`, `E`, `C`, `S`, and the addition
chemical potential for every particle number from zero through the model's
`N`. It propagates the first two derivatives of `log Z_N` through the
canonical recursion, rather than differentiating a sampled or tabulated
curve. Thermodynamic quantities require `beta > 0`; the canonical table and
the other finite-temperature weights remain defined at `beta == 0`.

`momentum_distribution` returns the energy, exact canonical occupation, and
occupation variance of every momentum mode. Its summary contains the
zero-momentum occupation and fraction, condensate density, kinetic energy,
and the finite-lattice second-moment coherence length

\[
\xi = \frac{1}{2\sin(\pi/L)}
\sqrt{\frac{n(0)}{\overline{n(q_{\min})}}-1}.
\]

The average in the denominator is over the `d` axial minimum nonzero momenta.
The length is zero when `L == 1` or the ratio is not positive. At `t == 0`,
all momentum states are degenerate, so the zero-momentum mode is not a unique
physical ground state.

`one_body_density_matrix` returns the translation-invariant
`<a_r^dagger a_(r+delta)>` at every torus displacement. It is evaluated from
normalized finite-torus kernels, without an open-world-line sector.

`exact_cycle_statistics` returns the expected cycle count, expected particles
in cycles, and random-particle cycle probability for each length. The sampled
counterpart is `sampled_cycle_histogram`; `longest_cycle_length` reports the
longest cycle of one configuration.

`log_canonical_partition_twisted` evaluates a canonical partition function
with arbitrary boundary twist. `twist_free_energy_curvature` evaluates the
zero-twist curvature analytically. With the twist convention used by the
dispersion,

\[
\frac{\partial^2 F}{\partial\phi_\alpha^2}\bigg|_0
=\frac{\langle W_\alpha^2\rangle}{\beta}.
\]

The raw curvature is exposed so callers can apply their preferred volume,
length, hopping, or dimensional normalization for a stiffness or superfluid
density.

## Equal-time configuration estimators

For workflows that evaluate several retained estimators on one ideal
configuration, `RetainedMeasurementContext` reduces the covering paths to
physical sites once and owns the matching retained-grid provenance. Both
`equal_time_observables` and `retained_density_correlations` accept this context;
their configuration-taking overloads construct a temporary context for one-off
use.

`equal_time_observables` averages over the `M` distinct exact retained slices
and returns:

- site density;
- the translation-averaged normal-ordered pair correlation `g2(delta)`;
- `S(q) = <n_q n_-q>/N`;
- the on-site occupation distribution;
- `<n^2>` and `<n(n-1)>` per site.

The ideal reference derivative of the Bose-Hubbard free energy is therefore

\[
\left.\frac{\partial F}{\partial U}\right|_{U=0}
=\frac{L^d}{2}\langle n(n-1)\rangle.
\]

## Retained imaginary-time grid

`retained_density_correlations` computes the connected density correlation
for every retained time lag and spatial displacement. It averages over all
retained time origins using periodic imaginary time. The corresponding
`retained_grid_matsubara_transform` uses

\[
\frac{\beta}{M}\sum_j\sum_r
e^{i\omega_n\tau_j-iq\cdot r}C_{nn}(r,\tau_j).
\]

This is the explicitly requested retained-grid transform. The sampled values
at every grid point are exact; replacing the continuous time integral with
the displayed sum is a measurement-grid approximation, not a Trotter
approximation to the path distribution.

The correlation result owns its inverse temperature, retained point count, and
torus layout as immutable grid provenance. The Matsubara transform consumes
that provenance directly rather than accepting a separately supplied model;
same-volume lattices with different dimensions and data with a different
inverse temperature therefore cannot be paired accidentally.

The transformed result is also valid by construction. Its `modes()` descriptor
owns the exact integer identities and physical frequencies, while `values()`
uses frequency-major then flat-momentum order and `at(frequency, momentum)`
checks both indices. Retained transforms always contain frequencies `0..M-1`
and every torus momentum in flat `TorusLayout` order. The shared
`MatsubaraModeSet`/`MatsubaraModeField` types can represent selected momenta and
signed frequencies for exact continuous-time measurements without introducing
a retained time count.

`retained_geometry_observables` additionally returns the same-label
covering-space mean-square displacement, torus return probability, and full
torus displacement distribution at each retained time. `retained_cycle_geometry`
returns the covering-space radius of gyration, maximum retained radius, cycle
length, and winding for every permutation cycle. These records support direct
cycle-length, winding, and geometry correlations.

For independent-sample workflows, `EqualTimeAccumulator`,
`RetainedDensityCorrelationAccumulator`, `RetainedGeometryAccumulator`,
`CycleStatisticsAccumulator`, and `WindingAccumulator` bind one retained grid
and particle count, reject incompatible samples before mutation, and own their
sample normalization. Cycle geometry means are conditioned on occurrences of
the corresponding cycle length. The cycle and winding accumulators return the
raw geometry and total winding from `observe()`, respectively, so a caller can
write per-sample traces without evaluating those quantities twice.

## Exact continuous-time Matsubara density

`ContinuousMeasurementContext` owns one continuous configuration's ordered
hopping geometry, while `ContinuousMatsubaraPlan` owns reusable phase data for
a selected `MatsubaraModeSet`. `continuous_particle_modes(context, plan)`
projects the exact, unnormalised density amplitude

\[
\rho_{\mathbf qn}=\int_0^\beta d\tau\,
e^{i\omega_n\tau}\sum_a e^{-i\mathbf q\cdot\mathbf x_a(\tau)}
\]

from piecewise-constant residence intervals; it does not introduce a retained
measurement grid.

`DensityMatsubaraAccumulator` binds the complete free `Model` and mode set,
rejects incompatible samples before mutation, and owns the sample count. For
the homogeneous fixed-`N` model it subtracts the exact amplitude mean
`beta*N` only at `(q,n)=(0,0)` and zero elsewhere. Its finished
`ContinuousMatsubaraDensityCorrelations` returns

\[
\chi_{nn}(\mathbf q,i\omega_n)
=\frac{1}{\beta V}\left\langle
|\rho_{\mathbf qn}-\beta N\delta_{\mathbf q,0}\delta_{n,0}|^2
\right\rangle.
\]

`mean_amplitude(frequency, momentum)` reports the sampled uncentred complex
mean as a symmetry diagnostic, while `at(frequency, momentum)` reports the
real nonnegative susceptibility. Both accessors use the selected mode set's
frequency-major ordering and check each index. The result retains its complete
model, modes, and nonzero sample count as provenance.

## Exact continuous-time on-site pair density

`continuous_pair_density_modes(context, plan)` reuses the same event geometry
and selected Matsubara modes to project the exact, unnormalised diagonal field

\[
P_{\mathbf qn}=\int_0^\beta d\tau\,
e^{i\omega_n\tau}\sum_{\mathbf x}
e^{-i\mathbf q\cdot\mathbf x}\binom{n_{\mathbf x}(\tau)}{2}.
\]

The projector replays a sparse physical-site occupancy map. It integrates the
current pair field over each positive-duration interval, then applies every hop
in a coincident event group before the next interval. Order-dependent
intermediate occupancies inside a zero-duration group are therefore never
measured. The `ContinuousPairDensityModes` result owns the complete free
`Model` and selected modes; `pair_density(frequency, momentum)` checks both
indices and has units of inverse energy.

The zero mode is exactly the existing interaction geometry:

\[
P_{\mathbf0,0}
=\texttt{pair_overlap_time(configuration)}.
\]

The result does not multiply by `U` and does not perform ensemble
normalization. Those choices belong to a later interaction-energy or
pair-density correlation workflow.

## Exact continuous-time hopping response

The same `ContinuousParticleModes` projection returns the dimensionless signed
hopping flux

\[
I_\alpha(\mathbf q,i\omega_n)=
\sum_{e:\alpha_e=\alpha}
s_e e^{i\omega_n\tau_e-i\mathbf q\cdot
(\mathbf x_e+s_e\hat\alpha/2)}
\]

and the unsigned event count `K_alpha` on each physical axis. The midpoint is
an oriented positive-bond Peierls-source convention; no electric charge or
lattice-spacing factor is included.

`HoppingResponseAccumulator` binds the complete free `Model` and mode set,
rejects incompatible samples before mutation, and uses the exact zero flux
mean of the source-free, time-reversal-invariant model. Its finished
`HoppingResponse` returns

\[
R_{\alpha\beta}=
\frac{\langle I_\alpha I_\beta^*\rangle}{\beta V},\qquad
D_\alpha=\frac{\langle K_\alpha\rangle}{\beta V},\qquad
\Lambda^p_{\alpha\beta}=\delta_{\alpha\beta}D_\alpha-R_{\alpha\beta}.
\]

`flux_response(...)` is the full gauge response, including the same-event
contact contribution; it is not the paramagnetic current correlation by
itself. `diamagnetic(axis)` returns `D`, and `paramagnetic(...)` derives
`Lambda^p` from the two stored authoritative terms. `mean_flux(...)` reports
the sampled uncentred complex amplitude as a symmetry diagnostic. The response
tensor is Hermitian by construction. `R`, `D`, and `Lambda^p` have units of
energy per site in the repository's `k_B=1`, unit-lattice-spacing convention.

## Exactness and checks

The canonical calculations are deterministic up to floating-point rounding.
Cycle winding uses the sampler's explicitly bounded tail tolerance. Density,
winding, and geometry averages are Monte Carlo estimates over independent
ideal configurations.

Useful exact checks are:

- `sum_q n(q) == N`;
- `L^d g1(0) == N` and `sum_delta g1(delta) == N_0`;
- `sum_l l <m_l> == N`;
- `sum_r <n_r> == N` and `S(0) == N`;
- the spatial sum of connected fixed-`N` density correlations is zero;
- every exact continuous-time `q == 0` density susceptibility is zero;
- `P_(0,0)` from continuous pair-density modes equals the exact integrated
  on-site pair count returned by `pair_overlap_time`;
- each retained displacement distribution sums to one for `N > 0`;
- twist curvature agrees with `<W_alpha^2>/beta`.
