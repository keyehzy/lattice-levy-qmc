from __future__ import annotations

import numpy as np

from interacting_lattice_levy import (
    ContinuousConfiguration,
    ContinuousPath,
    InteractingLatticeLevySampler,
    interaction_action,
    pair_overlap_time,
    resample_path_interval,
    sample_continuous_bridge_covering,
    sample_ideal_continuous_configuration,
    split_continuous_path,
)


def static_path(site: int, beta: float = 1.0) -> ContinuousPath:
    return ContinuousPath(
        duration=beta,
        start=np.array([site], dtype=np.int64),
        event_times=np.empty(0),
        jumps=np.empty((0, 1), dtype=np.int64),
        end=np.array([site], dtype=np.int64),
    )


def test_continuous_bridge_has_requested_endpoint():
    rng = np.random.default_rng(7)
    for _ in range(100):
        path = sample_continuous_bridge_covering(
            [0, -2], [5, 3], duration=1.4, t=0.8, rng=rng
        )
        assert np.array_equal(path.start, [0, -2])
        assert np.array_equal(path.end, [5, 3])
        assert np.array_equal(path.position_at(path.duration), path.end)
        assert np.all(path.event_times > 0.0)
        assert np.all(path.event_times < path.duration)


def test_split_path_is_continuous():
    rng = np.random.default_rng(8)
    path = sample_continuous_bridge_covering(
        [1], [8], duration=3.0, t=1.1, rng=rng
    )
    pieces = split_continuous_path(path, [0.7, 1.9])
    assert len(pieces) == 3
    assert np.array_equal(pieces[0].start, path.start)
    assert np.array_equal(pieces[-1].end, path.end)
    assert np.array_equal(pieces[0].end, pieces[1].start)
    assert np.array_equal(pieces[1].end, pieces[2].start)
    assert sum(piece.n_events for piece in pieces) == path.n_events


def test_pair_overlap_time_piecewise_exact():
    moving = ContinuousPath(
        duration=1.0,
        start=np.array([0]),
        event_times=np.array([0.25, 0.75]),
        jumps=np.array([[1], [-1]], dtype=np.int64),
        end=np.array([0]),
    )
    state = ContinuousConfiguration(
        L=3,
        d=1,
        N=2,
        beta=1.0,
        t=1.0,
        cycles=[(0,), (1,)],
        permutation=np.array([0, 1]),
        worldlines=[static_path(0), moving],
        log_Z0_N=0.0,
    )
    state.validate()
    assert np.isclose(pair_overlap_time(state), 0.5)
    assert np.isclose(interaction_action(state, U=3.0), 1.5)


def test_resampled_interval_preserves_outer_path_and_endpoints():
    rng = np.random.default_rng(10)
    path = sample_continuous_bridge_covering(
        [0, 0], [2, -1], duration=2.0, t=0.9, rng=rng
    )
    tau0, tau1 = 0.4, 1.6
    left = path.position_at(tau0)
    right = path.position_at(tau1)
    proposal = resample_path_interval(path, tau0, tau1, 0.9, rng)
    assert np.array_equal(proposal.start, path.start)
    assert np.array_equal(proposal.end, path.end)
    assert np.array_equal(proposal.position_at(tau0), left)
    assert np.array_equal(proposal.position_at(tau1), right)


def test_ideal_continuous_configuration_consistency():
    rng = np.random.default_rng(11)
    state = sample_ideal_continuous_configuration(
        N=8, beta=1.3, L=7, d=2, t=0.75, rng=rng
    )
    state.validate()
    assert sorted(state.permutation.tolist()) == list(range(8))
    assert sum(state.cycle_lengths) == 8
    assert state.total_winding().shape == (2,)


def test_u_zero_moves_are_rejection_free():
    sampler = InteractingLatticeLevySampler(
        N=5,
        beta=1.0,
        L=6,
        d=1,
        t=1.0,
        U=0.0,
        rng=np.random.default_rng(12),
    )
    for _ in range(20):
        assert sampler.segment_update(fraction=0.4)
        assert sampler.cycle_update()
        assert sampler.global_update()
    for stats in sampler.statistics.values():
        assert stats.acceptance == 1.0


def test_finite_u_updates_preserve_valid_configuration():
    sampler = InteractingLatticeLevySampler(
        N=6,
        beta=1.1,
        L=5,
        d=2,
        t=0.8,
        U=2.0,
        rng=np.random.default_rng(13),
    )
    for _ in range(20):
        sampler.sweep(
            segment_updates=6,
            segment_fraction=0.35,
            cycle_updates=2,
            global_updates=1,
        )
        sampler.state.validate()
        assert np.isclose(sampler.action, sampler.U * sampler.pair_overlap)
        assert sampler.pair_overlap >= 0.0


if __name__ == "__main__":
    tests = [
        test_continuous_bridge_has_requested_endpoint,
        test_split_path_is_continuous,
        test_pair_overlap_time_piecewise_exact,
        test_resampled_interval_preserves_outer_path_and_endpoints,
        test_ideal_continuous_configuration_consistency,
        test_u_zero_moves_are_rejection_free,
        test_finite_u_updates_preserve_valid_configuration,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
