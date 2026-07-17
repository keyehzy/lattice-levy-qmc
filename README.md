
# Lattice Lévy sampler for ideal Bose–Hubbard world lines

This implementation samples the canonical noninteracting Bose–Hubbard model

\[
H_0=-t\sum_{\langle ij\rangle}(a_i^\dagger a_j+a_j^\dagger a_i)
\]

on an \(L^d\) periodic hypercubic lattice.

## Included

- Exact free bridges on the covering lattice \(\mathbb Z^d\).
- Unequal recursive splitting, so the number of links need not be a power of two.
- A rejection-free midpoint draw based on conditional Poisson jump counts.
- Exact finite-torus one-particle traces from the momentum sum.
- Exact canonical cycle recursion in log space.
- Explicit winding-sector sampling.
- Conversion of long permutation cycles into labeled world lines.
- Verification tests, including a comparison against direct permutation enumeration.

The inversion samplers omit a tail whose relative weight is bounded by
`tail_tol` (default `1e-14`). Apart from floating-point arithmetic and this
chosen tolerance, the \(V=0\) skeleton sampler is exact.

## Install

```bash
python -m pip install numpy scipy matplotlib pytest
```

## Run tests

```bash
pytest -q
```

or

```bash
python test_lattice_levy.py
```

## Generate a 1D demonstration

```bash
python demo.py --N 4 --L 12 --M 64 --beta 1.0 --t 1.0
```

## Minimal use

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

print(cfg.permutation)
print([cycle.labels for cycle in cfg.cycles])
print([cycle.winding.tolist() for cycle in cfg.cycles])
print(cfg.worldlines.shape)  # (N, M+1, d)
```

For every particle label `i`,

```python
cfg.worldlines[i, -1] == cfg.worldlines[cfg.permutation[i], 0]
```

modulo the periodic lattice.

## Midpoint law

`sample_midpoint_covering_1d(a,b,tau_left,tau_right,t,rng)` samples

\[
P(k|a,b)=
\frac{
I_{|k-a|}(2t\tau_L)I_{|b-k|}(2t\tau_R)
}{
I_{|b-a|}(2t(\tau_L+\tau_R))
}.
\]

Internally it uses the equivalent continuous-time random-walk representation.
This avoids a fixed absolute spatial window, which fails for widely separated
endpoints.

## Finite-volume trace

The exact one-particle trace used for a cycle of duration \(s\) is

\[
Z_1(s)=
\left[
\sum_{n=0}^{L-1}
e^{2ts\cos(2\pi n/L)}
\right]^d.
\]

This includes all periodic images and winding sectors.
