
"""
Exact free-particle lattice bridges and canonical ideal-boson sampling.

Model
-----
    H0 = -t * sum_<ij> (|i><j| + |j><i|)

On the infinite 1D lattice,
    <i|exp(-tau H0)|j> = I_{j-i}(2 t tau).

The bridge sampler below is equivalent to sampling from
    P(k|a,b) = K(a,k;tau_L) K(k,b;tau_R) / K(a,b;tau_L+tau_R),
but uses a conditional continuous-time random-walk representation:

* N_+ and N_- are independent Poisson(t * tau) counts.
* Conditional on N_+ - N_- = b-a, the smaller count follows a
  Bessel distribution.
* Given total counts, jumps are assigned to the left subinterval by
  independent binomial draws.

This is rejection-free. The only controllable approximation is the omitted
tail in the one-dimensional inversion sampler, bounded by ``tail_tol``.

For a finite L^d torus, ideal-boson cycles are sampled with the exact
finite-volume one-particle traces. Winding sectors are sampled explicitly,
then each cycle is generated as a covering-space bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log, sqrt
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, ive, logsumexp


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CyclePath:
    """One permutation cycle and its sampled long world line."""

    labels: tuple[int, ...]
    base_point: IntArray
    winding: IntArray
    covering_path: IntArray
    torus_path: IntArray


@dataclass(frozen=True)
class IdealBosonConfiguration:
    """A canonical ideal-boson world-line configuration."""

    L: int
    d: int
    N: int
    M: int
    beta: float
    t: float
    cycles: tuple[CyclePath, ...]
    permutation: IntArray
    worldlines: IntArray
    worldlines_covering: IntArray
    log_ZN: float


def _validate_positive_time(tau: float, name: str = "tau") -> None:
    if not np.isfinite(tau) or tau < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {tau!r}")


def _sample_bessel_pair_count(
    abs_delta: int,
    lam: float,
    rng: np.random.Generator,
    *,
    tail_tol: float = 1e-14,
    max_terms: int = 2_000_000,
) -> int:
    r"""
    Sample m >= 0 from

        p(m | D) ∝ lam^(2m+D) / [m! (m+D)!],  D = abs_delta.

    This is the conditional distribution of the smaller of two independent
    Poisson(lam) variables given that their difference has magnitude D.

    The omitted right tail is bounded by ``tail_tol``. The ratio

        u_{m+1}/u_m = lam^2 / [(m+1)(m+D+1)]

    decreases monotonically, giving a geometric tail bound.
    """
    D = int(abs_delta)
    if D < 0:
        raise ValueError("abs_delta must be nonnegative")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")
    if not (0 < tail_tol < 1):
        raise ValueError("tail_tol must lie in (0, 1)")

    if lam == 0.0:
        if D != 0:
            raise ValueError(
                "A nonzero endpoint displacement has zero weight when t*tau = 0."
            )
        return 0

    # The mode is near the positive root of m(m+D)=lam^2.
    mode_est = max(0, int(floor(0.5 * (sqrt(D * D + 4.0 * lam * lam) - D))))
    M = max(16, mode_est + int(ceil(12.0 * sqrt(lam + 1.0))) + 8)

    log_lam = log(lam)
    log_tol = log(tail_tol)

    while True:
        if M > max_terms:
            raise RuntimeError(
                f"Bessel-count support exceeded max_terms={max_terms}. "
                "Increase max_terms or use a coarser bridge hierarchy."
            )

        m = np.arange(M + 1, dtype=np.float64)
        log_u = (
            (2.0 * m + D) * log_lam
            - gammaln(m + 1.0)
            - gammaln(m + D + 1.0)
        )
        log_inside = float(logsumexp(log_u))

        m1 = M + 1
        log_u1 = (
            (2.0 * m1 + D) * log_lam
            - float(gammaln(m1 + 1.0))
            - float(gammaln(m1 + D + 1.0))
        )
        # Ratio u_{M+2}/u_{M+1}; all subsequent ratios are smaller.
        ratio_after_first = lam * lam / ((M + 2.0) * (M + D + 2.0))

        if ratio_after_first >= 1.0:
            M = max(M + 32, 2 * M)
            continue

        log_tail_bound = log_u1 - np.log1p(-ratio_after_first)
        if log_tail_bound - log_inside <= log_tol:
            probs = np.exp(log_u - log_inside)
            cdf = np.cumsum(probs)
            # The CDF ends below 1 only by the controlled omitted tail.
            u = rng.random() * cdf[-1]
            return int(np.searchsorted(cdf, u, side="right"))

        M = max(M + 32, int(1.5 * M) + 1)


def sample_midpoint_covering_1d(
    a: int,
    b: int,
    tau_left: float,
    tau_right: float,
    t: float,
    rng: np.random.Generator,
    *,
    tail_tol: float = 1e-14,
) -> int:
    r"""
    Sample the exact free midpoint on the covering lattice Z.

    The target law is

        P(k|a,b) =
          I_|k-a|(2 t tau_left) I_|b-k|(2 t tau_right)
          ------------------------------------------------
          I_|b-a|(2 t (tau_left + tau_right)).

    Sampling is performed through conditional Poisson jump counts rather than
    by directly truncating k.
    """
    _validate_positive_time(tau_left, "tau_left")
    _validate_positive_time(tau_right, "tau_right")
    if not np.isfinite(t) or t < 0:
        raise ValueError("t must be finite and nonnegative")

    total = tau_left + tau_right
    if total == 0.0:
        if a != b:
            raise ValueError("Endpoints must coincide for a zero-duration bridge.")
        return int(a)
    if tau_left == 0.0:
        return int(a)
    if tau_right == 0.0:
        return int(b)

    delta = int(b) - int(a)
    lam = t * total

    if lam == 0.0:
        if delta != 0:
            raise ValueError("The requested bridge has zero free-particle weight.")
        return int(a)

    m = _sample_bessel_pair_count(
        abs(delta), lam, rng, tail_tol=tail_tol
    )

    if delta >= 0:
        n_plus = m + delta
        n_minus = m
    else:
        n_plus = m
        n_minus = m - delta

    frac_left = tau_left / total
    plus_left = int(rng.binomial(n_plus, frac_left))
    minus_left = int(rng.binomial(n_minus, frac_left))
    return int(a) + plus_left - minus_left


def sample_midpoint_covering(
    a: Sequence[int] | IntArray,
    b: Sequence[int] | IntArray,
    tau_left: float,
    tau_right: float,
    t: float,
    rng: np.random.Generator,
    *,
    tail_tol: float = 1e-14,
) -> IntArray:
    """Coordinate-factorized d-dimensional covering-space midpoint."""
    aa = np.asarray(a, dtype=np.int64)
    bb = np.asarray(b, dtype=np.int64)
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError("a and b must be one-dimensional arrays of equal shape")

    out = np.empty_like(aa)
    for alpha in range(aa.size):
        out[alpha] = sample_midpoint_covering_1d(
            int(aa[alpha]),
            int(bb[alpha]),
            tau_left,
            tau_right,
            t,
            rng,
            tail_tol=tail_tol,
        )
    return out


def sample_bridge_covering(
    a: Sequence[int] | IntArray,
    b: Sequence[int] | IntArray,
    total_time: float,
    n_steps: int,
    t: float,
    rng: np.random.Generator,
    *,
    tail_tol: float = 1e-14,
) -> IntArray:
    """
    Sample a free bridge on Z^d at ``n_steps + 1`` equally spaced times.

    Unequal recursive splits are used when ``n_steps`` is not a power of two.
    """
    _validate_positive_time(total_time, "total_time")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    aa = np.asarray(a, dtype=np.int64)
    bb = np.asarray(b, dtype=np.int64)
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError("a and b must be one-dimensional arrays of equal shape")
    if total_time == 0.0 and not np.array_equal(aa, bb):
        raise ValueError("Endpoints must coincide for a zero-duration bridge.")

    path = np.empty((n_steps + 1, aa.size), dtype=np.int64)
    path[0] = aa
    path[-1] = bb
    dt = total_time / n_steps

    stack: list[tuple[int, int]] = [(0, n_steps)]
    while stack:
        left_idx, right_idx = stack.pop()
        width = right_idx - left_idx
        if width <= 1:
            continue

        mid_idx = left_idx + width // 2
        tau_left = (mid_idx - left_idx) * dt
        tau_right = (right_idx - mid_idx) * dt

        path[mid_idx] = sample_midpoint_covering(
            path[left_idx],
            path[right_idx],
            tau_left,
            tau_right,
            t,
            rng,
            tail_tol=tail_tol,
        )
        stack.append((mid_idx, right_idx))
        stack.append((left_idx, mid_idx))

    return path


def periodic_kernel_scaled_1d(
    delta: int,
    tau: float,
    L: int,
    t: float,
) -> float:
    r"""
    Return exp(-2 t tau) K_L(delta; tau) on a periodic ring.

    The scaled finite momentum sum is stable for large positive ``t*tau``:

        exp(-2t tau) K_L =
          (1/L) sum_q exp[2t tau (cos q - 1)] cos(q delta).
    """
    _validate_positive_time(tau)
    if L < 1:
        raise ValueError("L must be positive")
    if t < 0 or not np.isfinite(t):
        raise ValueError("t must be finite and nonnegative")

    q = 2.0 * np.pi * np.arange(L, dtype=np.float64) / L
    terms = np.exp(2.0 * t * tau * (np.cos(q) - 1.0)) * np.cos(q * delta)
    value = float(np.mean(terms))
    # Roundoff can produce tiny negative values for a positive kernel.
    return max(0.0, value)


def sample_midpoint_torus_1d(
    a: int,
    b: int,
    tau_left: float,
    tau_right: float,
    L: int,
    t: float,
    rng: np.random.Generator,
) -> int:
    """
    Sample a midpoint directly on a finite periodic ring.

    This sums all winding sectors implicitly. Use the covering-space sampler
    instead when winding observables are required.
    """
    sites = np.arange(L, dtype=np.int64)
    left = np.array(
        [periodic_kernel_scaled_1d(int(k - a), tau_left, L, t) for k in sites]
    )
    right = np.array(
        [periodic_kernel_scaled_1d(int(b - k), tau_right, L, t) for k in sites]
    )
    weights = left * right
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Failed to normalize periodic midpoint weights")
    cdf = np.cumsum(weights / total)
    return int(np.searchsorted(cdf, rng.random(), side="right"))


def log_one_particle_trace_torus(
    duration: float,
    L: int,
    d: int,
    t: float,
) -> float:
    r"""
    Exact log Z_1(duration) on an L^d periodic hypercubic lattice.

        Z_1(s) = [sum_{n=0}^{L-1} exp(2 t s cos(2 pi n/L))]^d.
    """
    _validate_positive_time(duration, "duration")
    if L < 1 or d < 1:
        raise ValueError("L and d must be positive")
    if t < 0 or not np.isfinite(t):
        raise ValueError("t must be finite and nonnegative")

    q = 2.0 * np.pi * np.arange(L, dtype=np.float64) / L
    log_trace_1d = float(logsumexp(2.0 * t * duration * np.cos(q)))
    return d * log_trace_1d


def canonical_log_partition(
    N: int,
    beta: float,
    L: int,
    d: int,
    t: float,
) -> tuple[FloatArray, FloatArray]:
    r"""
    Compute ideal-boson cycle weights and canonical partition functions.

    Returns
    -------
    log_z:
        ``log_z[l] = log Z_1(l beta)`` for l=1,...,N.
    log_Z:
        ``log_Z[n] = log Z_n`` for n=0,...,N, using
            Z_n = (1/n) sum_{l=1}^n z_l Z_{n-l}.
    """
    if N < 0:
        raise ValueError("N must be nonnegative")
    _validate_positive_time(beta, "beta")
    if beta == 0.0:
        # The formulas remain well-defined, but an explicit zero-temperature
        # interval is usually an input error.
        pass

    log_z = np.full(N + 1, -np.inf, dtype=np.float64)
    for ell in range(1, N + 1):
        log_z[ell] = log_one_particle_trace_torus(
            ell * beta, L=L, d=d, t=t
        )

    log_Z = np.full(N + 1, -np.inf, dtype=np.float64)
    log_Z[0] = 0.0
    for n in range(1, N + 1):
        terms = np.array(
            [log_z[ell] + log_Z[n - ell] for ell in range(1, n + 1)],
            dtype=np.float64,
        )
        log_Z[n] = float(logsumexp(terms) - np.log(n))

    return log_z, log_Z


def sample_cycle_labels(
    N: int,
    log_z: FloatArray,
    log_Z: FloatArray,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """
    Sample a labeled permutation by the exact canonical cycle recursion.
    """
    if N == 0:
        return []
    if len(log_z) < N + 1 or len(log_Z) < N + 1:
        raise ValueError("log_z and log_Z are too short")

    remaining = list(range(N))
    cycles: list[tuple[int, ...]] = []

    while remaining:
        n = len(remaining)
        log_probs = np.array(
            [
                log_z[ell] + log_Z[n - ell] - np.log(n) - log_Z[n]
                for ell in range(1, n + 1)
            ],
            dtype=np.float64,
        )
        probs = np.exp(log_probs - logsumexp(log_probs))
        ell = int(rng.choice(np.arange(1, n + 1), p=probs))

        distinguished = remaining[0]
        if ell == 1:
            cycle = (distinguished,)
        else:
            pool = np.asarray(remaining[1:], dtype=np.int64)
            others = rng.choice(pool, size=ell - 1, replace=False)
            others = np.asarray(others, dtype=np.int64)
            rng.shuffle(others)
            cycle = tuple([distinguished, *map(int, others)])

        chosen = set(cycle)
        remaining = [label for label in remaining if label not in chosen]
        cycles.append(cycle)

    return cycles


def sample_winding_1d(
    L: int,
    duration: float,
    t: float,
    rng: np.random.Generator,
    *,
    tail_tol: float = 1e-14,
    max_winding: int = 1_000_000,
) -> int:
    r"""
    Sample the exact winding sector for a closed path on a ring.

        P(w) ∝ I_{|w|L}(2 t duration),  w in Z.

    ``ive`` is used because the common exp(2 t duration) factor cancels.
    """
    if L < 1:
        raise ValueError("L must be positive")
    _validate_positive_time(duration, "duration")
    if t < 0 or not np.isfinite(t):
        raise ValueError("t must be finite and nonnegative")
    if not (0 < tail_tol < 1):
        raise ValueError("tail_tol must lie in (0, 1)")

    z = 2.0 * t * duration
    if z == 0.0:
        return 0

    w_max = max(4, int(ceil(8.0 * sqrt(max(z, 1.0)) / L)) + 4)
    log_tol = log(tail_tol)

    while True:
        if w_max > max_winding:
            raise RuntimeError("Winding support exceeded max_winding")

        orders = L * np.arange(w_max + 1, dtype=np.int64)
        nonnegative_weights = np.asarray(ive(orders, z), dtype=np.float64)
        nonnegative_weights = np.maximum(nonnegative_weights, 0.0)

        included = (
            nonnegative_weights[0]
            + 2.0 * float(nonnegative_weights[1:].sum())
        )
        if included <= 0.0 or not np.isfinite(included):
            raise FloatingPointError("Failed to evaluate winding weights")

        first_omitted = float(ive((w_max + 1) * L, z))
        second_omitted = float(ive((w_max + 2) * L, z))
        first_omitted = max(0.0, first_omitted)
        second_omitted = max(0.0, second_omitted)

        if first_omitted == 0.0:
            break

        ratio = second_omitted / first_omitted
        if ratio < 1.0:
            tail_bound = 2.0 * first_omitted / (1.0 - ratio)
            if log(tail_bound) - log(included) <= log_tol:
                break

        w_max = max(w_max + 8, int(1.5 * w_max) + 1)

    windings = np.arange(-w_max, w_max + 1, dtype=np.int64)
    weights = np.asarray(ive(np.abs(windings) * L, z), dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    probs = weights / weights.sum()
    return int(rng.choice(windings, p=probs))


def sample_ideal_boson_configuration(
    N: int,
    beta: float,
    M: int,
    L: int,
    d: int,
    t: float,
    rng: np.random.Generator | None = None,
    *,
    tail_tol: float = 1e-14,
) -> IdealBosonConfiguration:
    """
    Sample an exact canonical ideal-boson skeleton on an L^d torus.

    Parameters
    ----------
    N:
        Number of bosons.
    beta:
        Inverse temperature.
    M:
        Number of retained time links per interval beta.
    L, d:
        Linear torus size and spatial dimension.
    t:
        Nearest-neighbor hopping amplitude.
    rng:
        NumPy random generator.
    tail_tol:
        Relative bound on omitted inversion tails.

    Returns
    -------
    IdealBosonConfiguration
        Includes cycle decomposition, winding vectors, long cycle paths,
        labeled world lines, and the sampled permutation.
    """
    if N < 0:
        raise ValueError("N must be nonnegative")
    if M < 1 or L < 1 or d < 1:
        raise ValueError("M, L, and d must be positive")
    _validate_positive_time(beta, "beta")
    if t < 0 or not np.isfinite(t):
        raise ValueError("t must be finite and nonnegative")
    if rng is None:
        rng = np.random.default_rng()

    log_z, log_Z = canonical_log_partition(N, beta, L, d, t)
    cycle_labels = sample_cycle_labels(N, log_z, log_Z, rng)

    permutation = np.empty(N, dtype=np.int64)
    worldlines = np.empty((N, M + 1, d), dtype=np.int64)
    worldlines_covering = np.empty((N, M + 1, d), dtype=np.int64)
    cycle_paths: list[CyclePath] = []

    for labels in cycle_labels:
        ell = len(labels)
        duration = ell * beta
        base = rng.integers(0, L, size=d, dtype=np.int64)
        winding = np.array(
            [
                sample_winding_1d(
                    L, duration, t, rng, tail_tol=tail_tol
                )
                for _ in range(d)
            ],
            dtype=np.int64,
        )
        endpoint = base + L * winding

        covering = sample_bridge_covering(
            base,
            endpoint,
            total_time=duration,
            n_steps=ell * M,
            t=t,
            rng=rng,
            tail_tol=tail_tol,
        )
        torus = np.mod(covering, L)

        for j, label in enumerate(labels):
            start = j * M
            stop = (j + 1) * M
            worldlines[label] = torus[start : stop + 1]
            worldlines_covering[label] = covering[start : stop + 1]
            permutation[label] = labels[(j + 1) % ell]

        cycle_paths.append(
            CyclePath(
                labels=labels,
                base_point=base.copy(),
                winding=winding.copy(),
                covering_path=covering,
                torus_path=torus,
            )
        )

    return IdealBosonConfiguration(
        L=L,
        d=d,
        N=N,
        M=M,
        beta=beta,
        t=t,
        cycles=tuple(cycle_paths),
        permutation=permutation,
        worldlines=worldlines,
        worldlines_covering=worldlines_covering,
        log_ZN=float(log_Z[N]),
    )


def exact_midpoint_pmf_window(
    a: int,
    b: int,
    tau_left: float,
    tau_right: float,
    t: float,
    k_values: Iterable[int],
) -> FloatArray:
    """
    Evaluate the infinite-lattice bridge PMF on a supplied finite set.

    The returned values are normalized by the exact Bessel denominator, so
    their sum is the exact probability mass contained in ``k_values``.
    """
    ks = np.asarray(list(k_values), dtype=np.int64)
    z_left = 2.0 * t * tau_left
    z_right = 2.0 * t * tau_right
    z_total = z_left + z_right

    if z_total == 0.0:
        return (ks == a).astype(np.float64) if a == b else np.zeros_like(ks, dtype=float)

    num = ive(np.abs(ks - a), z_left) * ive(np.abs(b - ks), z_right)
    den = float(ive(abs(b - a), z_total))
    if den <= 0.0:
        raise FloatingPointError("Bessel denominator underflowed")
    # Scaling factors cancel because z_total=z_left+z_right.
    return np.asarray(num / den, dtype=np.float64)


__all__ = [
    "CyclePath",
    "IdealBosonConfiguration",
    "sample_midpoint_covering_1d",
    "sample_midpoint_covering",
    "sample_bridge_covering",
    "periodic_kernel_scaled_1d",
    "sample_midpoint_torus_1d",
    "log_one_particle_trace_torus",
    "canonical_log_partition",
    "sample_cycle_labels",
    "sample_winding_1d",
    "sample_ideal_boson_configuration",
    "exact_midpoint_pmf_window",
]
