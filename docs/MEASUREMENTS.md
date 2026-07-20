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

`retained_geometry_observables` additionally returns the same-label
covering-space mean-square displacement, torus return probability, and full
torus displacement distribution at each retained time. `retained_cycle_geometry`
returns the covering-space radius of gyration, maximum retained radius, cycle
length, and winding for every permutation cycle. These records support direct
cycle-length, winding, and geometry correlations.

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
- each retained displacement distribution sums to one for `N > 0`;
- twist curvature agrees with `<W_alpha^2>/beta`.
