# Canonical lattice Lévy QMC for Bose–Hubbard world lines

This repository is a compact reference implementation of canonical-ensemble
world-line sampling for bosons on a periodic hypercubic lattice. It contains
two related samplers:

| Module | Target | Path representation | Role |
| --- | --- | --- | --- |
| `lattice_levy.py` | Ideal gas, \(U=0\) | Positions at \(M+1\) times per particle | Exact ideal-gas samples and visualization |
| `interacting_lattice_levy.py` | Finite \(U\) | Continuous-time nearest-neighbor jump events | Metropolis Monte Carlo for the Bose–Hubbard model |

The code is deliberately small and explicit. [Architecture and
algorithms](docs/ARCHITECTURE.md) explains how it works; [C++ porting
guide](docs/CPP_PORT.md) records the data structures, interfaces, invariants,
and numerical details that a rewrite should preserve.

## Model

At fixed particle number \(N\), the Hamiltonian is

\[
H=-t\sum_{\langle ij\rangle}
  (a_i^\dagger a_j+a_j^\dagger a_i)
  +\frac{U}{2}\sum_i n_i(n_i-1)
\]

on an \(L^d\) periodic hypercubic lattice. The ideal sampler draws directly
from the \(U=0\) canonical measure. The interacting sampler uses that measure
for proposals and accepts a proposed path configuration \(R'\) with

\[
\min\left(1,\exp[-S_U(R')+S_U(R)]\right),\qquad
S_U(R)=U\int_0^\beta d\tau\sum_x\frac{n_x(\tau)(n_x(\tau)-1)}{2}.
\]

Free bridges are conditioned continuous-time random walks. The interacting
algorithm therefore has no Trotter time discretization. In the ideal skeleton
sampler, `M` selects which points of an exact bridge are retained; it is an
output resolution, not a Trotter approximation.

The discrete inversion samplers omit a tail whose relative weight is bounded
by `tail_tol` (default `1e-14`). Subject to that tolerance and floating-point
roundoff, the free proposals are exact. Finite-\(U\) results are still Monte
Carlo estimates and require equilibration, autocorrelation analysis, and
statistical error bars.

## Repository map

| Path | Purpose |
| --- | --- |
| `lattice_levy.py` | Free bridge primitives, torus traces, canonical cycle recursion, windings, and ideal skeleton assembly |
| `interacting_lattice_levy.py` | Event-driven paths, interaction action, update moves, observables, and the finite-\(U\) sampler |
| `demo.py` | One-dimensional ideal-world-line plot |
| `demo_interacting.py` | Finite-\(U\) trace and acceptance-rate demo |
| `validate_interacting_ed.py` | Small-system comparison with exact diagonalization |
| `test_lattice_levy.py` | Free-kernel, recursion, winding, bridge, and configuration tests |
| `test_interacting_lattice_levy.py` | Continuous-path, action, update, and state-invariant tests |
| `interacting_ed_validation.txt` | Checked-in reference output from the exact-diagonalization comparison |

## Install and verify

Python 3.10 or newer is recommended. Install the numerical and test
dependencies in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
```

The tests include empirical sampling checks, so their exact run time depends
on the machine. The checked-in validation can be regenerated with:

```bash
python validate_interacting_ed.py
```

That program compares the Monte Carlo total, kinetic, and interaction
energies and pair occupancy with exact diagonalization for \(N=2\), \(L=3\).
Agreement is statistical rather than bit-for-bit.

## Ideal-gas use

```python
import numpy as np
from lattice_levy import sample_ideal_boson_configuration

rng = np.random.default_rng(2026)
cfg = sample_ideal_boson_configuration(
    N=8,
    beta=2.0,
    M=64,
    L=16,
    d=2,
    t=1.0,
    rng=rng,
)

print(cfg.log_ZN)
print(cfg.permutation)
print([cycle.labels for cycle in cfg.cycles])
print([cycle.winding.tolist() for cycle in cfg.cycles])
print(cfg.worldlines.shape)  # (N, M+1, d), reduced modulo L
```

`cfg.worldlines_covering` contains the corresponding unwrapped coordinates.
For every particle label `i`, the endpoint joins the next permutation label:

```python
j = cfg.permutation[i]
assert np.array_equal(cfg.worldlines[i, -1], cfg.worldlines[j, 0])
```

Generate the checked-in style of one-dimensional plot with:

```bash
python demo.py --N 4 --L 12 --M 64 --beta 1.0 --t 1.0
```

## Interacting use

```python
import numpy as np
from interacting_lattice_levy import InteractingLatticeLevySampler

sampler = InteractingLatticeLevySampler(
    N=6,
    beta=1.5,
    L=8,
    d=1,
    t=1.0,
    U=2.0,
    rng=np.random.default_rng(20260717),
)

# Equilibrate. Global moves are what change the permutation-cycle structure.
for _ in range(500):
    sampler.sweep(
        segment_updates=sampler.N,
        segment_fraction=0.35,
        cycle_updates=1,
        global_updates=1,
    )

samples = sampler.run(
    2_000,
    thin=2,
    segment_fraction=0.35,
    cycle_updates=1,
    global_updates=1,
)
print(samples[-1])
print({name: stat.acceptance for name, stat in sampler.statistics.items()})
```

`observables()` and `run()` return:

| Key | Meaning |
| --- | --- |
| `action` | \(U\) times the integrated on-site pair count |
| `pair_overlap_time` | \(\int d\tau\sum_x n_x(n_x-1)/2\) |
| `double_occupancy_per_site` | Pair-overlap time divided by \(\beta L^d\) |
| `kinetic_energy` | Continuous-time expansion estimator \(-K/\beta\), with \(K\) the total jump count |
| `interaction_energy` | \(U\) times pair-overlap time divided by \(\beta\) |
| `total_energy` | Sum of the two energy estimators |
| `n_events` | Total number of hopping events in the configuration |
| `winding` | Total covering-space displacement divided by \(L\) |
| `cycle_lengths` | Current permutation-cycle lengths |

The demonstration driver runs the same workflow and plots running means:

```bash
python demo_interacting.py --N 6 --L 8 --beta 1.5 --t 1.0 --U 2.0
```

## Important conventions

- Particle labels index world-line segments of duration `beta`; permutation
  cycles connect their endpoints into longer bosonic loops.
- Covering-space coordinates are signed integers and may lie outside
  `[0, L)`. Physical positions use Euclidean reduction modulo `L`.
- `ContinuousPath` is piecewise constant and right-continuous: an event at
  time `tau` is included in `position_at(tau)`.
- Segment moves preserve fixed covering-space endpoints. Cycle moves preserve
  cycle labels but can change their geometry and winding. Only global moves
  replace the permutation-cycle structure.
- The implementation is canonical. A chemical-potential contribution is
  constant at fixed `N` and is not part of Metropolis decisions.
- The same `numpy.random.Generator` drives every random decision, making a
  Python run reproducible for a fixed seed and dependency stack.

## Scope and current limitations

This is a research/reference implementation, not a tuned production engine.
It assumes one hopping amplitude and one length for every dimension. The
interaction overlap is recomputed for the whole configuration after each
proposal, which keeps the detailed-balance logic transparent but costs
\(O(E\log E)\) for \(E\) jump events. Global ideal-gas proposals guarantee
access to different permutation sectors, but their acceptance can become poor
for larger systems or stronger interactions. Checkpointing, parallel chains,
automatic error analysis, and production diagnostics are outside the current
scope.
