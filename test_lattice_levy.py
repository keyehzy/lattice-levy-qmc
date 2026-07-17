
from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
from scipy.special import iv

from lattice_levy import (
    canonical_log_partition,
    exact_midpoint_pmf_window,
    log_one_particle_trace_torus,
    sample_bridge_covering,
    sample_ideal_boson_configuration,
    sample_midpoint_covering_1d,
)


def cycle_lengths_of_permutation(p):
    n = len(p)
    seen = [False] * n
    out = []
    for i in range(n):
        if seen[i]:
            continue
        j = i
        ell = 0
        while not seen[j]:
            seen[j] = True
            ell += 1
            j = p[j]
        out.append(ell)
    return out


def test_bessel_convolution():
    a, b = 0, 3
    z1, z2 = 1.3, 2.1
    ks = np.arange(-80, 81)
    lhs = np.sum(iv(np.abs(ks - a), z1) * iv(np.abs(b - ks), z2))
    rhs = iv(abs(b - a), z1 + z2)
    assert np.isclose(lhs, rhs, rtol=5e-14, atol=5e-14)


def test_midpoint_sampler_empirical():
    rng = np.random.default_rng(12345)
    a, b = 0, 2
    tau_left = tau_right = 0.5
    t = 1.0
    ks = np.arange(-8, 11)
    exact = exact_midpoint_pmf_window(
        a, b, tau_left, tau_right, t, ks
    )
    assert exact.sum() > 1.0 - 1e-13

    n = 100_000
    samples = np.array(
        [
            sample_midpoint_covering_1d(
                a, b, tau_left, tau_right, t, rng
            )
            for _ in range(n)
        ]
    )
    counts = Counter(samples.tolist())
    empirical = np.array([counts[int(k)] / n for k in ks])
    # Loose enough to avoid flaky failures, tight enough to catch a wrong law.
    assert np.max(np.abs(empirical - exact)) < 0.006


def test_widely_separated_endpoints():
    rng = np.random.default_rng(9)
    samples = np.array(
        [
            sample_midpoint_covering_1d(0, 100, 0.5, 0.5, 10.0, rng)
            for _ in range(20_000)
        ]
    )
    # The midpoint is centered near 50; this catches the bad absolute [-30,30] rule.
    assert abs(samples.mean() - 50.0) < 0.25
    assert samples.std() > 3.0


def test_finite_torus_trace_against_momentum_sum():
    L, d, t, s = 5, 2, 0.7, 1.3
    q = 2.0 * np.pi * np.arange(L) / L
    direct_1d = np.sum(np.exp(2.0 * t * s * np.cos(q)))
    direct = direct_1d**d
    got = math.exp(log_one_particle_trace_torus(s, L, d, t))
    assert np.isclose(got, direct, rtol=2e-14, atol=2e-14)


def test_canonical_recursion_against_permutation_enumeration():
    N, beta, L, d, t = 5, 0.8, 4, 1, 0.9
    log_z, log_Z = canonical_log_partition(N, beta, L, d, t)
    z = np.exp(log_z)

    total = 0.0
    for p in itertools.permutations(range(N)):
        weight = 1.0
        for ell in cycle_lengths_of_permutation(p):
            weight *= z[ell]
        total += weight
    total /= math.factorial(N)

    assert np.isclose(math.exp(log_Z[N]), total, rtol=2e-13, atol=2e-13)


def test_bridge_grid_for_non_power_of_two():
    rng = np.random.default_rng(123)
    path = sample_bridge_covering(
        [0, 0], [3, -2], total_time=1.7, n_steps=15, t=0.8, rng=rng
    )
    assert path.shape == (16, 2)
    assert np.array_equal(path[0], [0, 0])
    assert np.array_equal(path[-1], [3, -2])


def test_ideal_boson_configuration_consistency():
    rng = np.random.default_rng(2026)
    cfg = sample_ideal_boson_configuration(
        N=7, beta=1.2, M=7, L=6, d=2, t=0.9, rng=rng
    )

    assert cfg.worldlines.shape == (7, 8, 2)
    assert sorted(cfg.permutation.tolist()) == list(range(7))

    for i in range(cfg.N):
        j = int(cfg.permutation[i])
        assert np.array_equal(cfg.worldlines[i, -1], cfg.worldlines[j, 0])

    labels = sorted(label for cyc in cfg.cycles for label in cyc.labels)
    assert labels == list(range(cfg.N))

    for cyc in cfg.cycles:
        displacement = cyc.covering_path[-1] - cyc.covering_path[0]
        assert np.array_equal(displacement, cfg.L * cyc.winding)
        assert np.array_equal(cyc.torus_path[0], cyc.torus_path[-1])


if __name__ == "__main__":
    tests = [
        test_bessel_convolution,
        test_midpoint_sampler_empirical,
        test_widely_separated_endpoints,
        test_finite_torus_trace_against_momentum_sum,
        test_canonical_recursion_against_permutation_enumeration,
        test_bridge_grid_for_non_power_of_two,
        test_ideal_boson_configuration_consistency,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
