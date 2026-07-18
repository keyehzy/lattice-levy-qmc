"""
Exact continuous-time lattice-Lévy Monte Carlo for the canonical Bose-Hubbard model.

The target measure is

    d pi_U[R] proportional to d pi_0[R] exp(-S_U[R]),

where ``pi_0`` is the exact canonical ideal-boson random-walk loop measure and

    S_U[R] = U * integral_0^beta d tau sum_x n_x(tau)(n_x(tau)-1)/2.

Free proposals are sampled exactly as conditioned continuous-time random walks.
For a fixed bridge endpoint in one coordinate, the positive and negative jump
counts are independent Poisson(t*T) variables conditioned on their difference.
Jump times are then uniform order statistics.  Consequently, segment updates,
whole-cycle updates, and global ideal-gas independence proposals have the exact
Metropolis ratio

    min(1, exp[-Delta S_U]).

Random-seam stitch updates additionally heat-bath sample a local endpoint
matching with exact torus kernels before drawing the bridges.  They obey the
same action-only ratio while changing permutation topology through closed
successor transpositions.  A sparse occupancy ledger evaluates local action
changes without rescanning the full system.

There is no Trotter discretization.  ``tail_tol`` only controls the numerically
omitted tail in discrete inversion samplers for conditioned jump counts and
winding sectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import ceil, log, sqrt
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import ive

from lattice_levy import (
    _sample_bessel_pair_count,
    canonical_log_partition,
    sample_cycle_labels,
    sample_winding_1d,
)


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def _as_int_vector(value: Sequence[int] | IntArray, *, name: str) -> IntArray:
    out = np.asarray(value, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    return out.copy()


def _validate_model(beta: float, L: int, d: int, t: float, U: float) -> None:
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    if L < 1 or d < 1:
        raise ValueError("L and d must be positive")
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("t must be finite and nonnegative")
    if not np.isfinite(U):
        raise ValueError("U must be finite")


@dataclass(frozen=True)
class ContinuousPath:
    """Piecewise-constant covering-space path on ``[0, duration]``.

    ``event_times[k]`` is the time of ``jumps[k]``. Each jump is a nearest-
    neighbor unit vector. Positions are right-continuous at event times.
    """

    duration: float
    start: IntArray
    event_times: FloatArray
    jumps: IntArray
    end: IntArray

    def __post_init__(self) -> None:
        start = _as_int_vector(self.start, name="start")
        end = _as_int_vector(self.end, name="end")
        times = np.asarray(self.event_times, dtype=np.float64).copy()
        jumps = np.asarray(self.jumps, dtype=np.int64).copy()

        if not np.isfinite(self.duration) or self.duration < 0.0:
            raise ValueError("duration must be finite and nonnegative")
        if start.shape != end.shape:
            raise ValueError("start and end must have equal dimension")
        if times.ndim != 1:
            raise ValueError("event_times must be one-dimensional")
        if jumps.shape != (times.size, start.size):
            raise ValueError("jumps must have shape (number_of_events, dimension)")
        if np.any(~np.isfinite(times)):
            raise ValueError("event times must be finite")
        if np.any(times < 0.0) or np.any(times > self.duration):
            raise ValueError("event times must lie in [0, duration]")
        if times.size > 1 and np.any(times[1:] < times[:-1]):
            raise ValueError("event_times must be sorted")

        if jumps.size:
            abs_jumps = np.abs(jumps)
            if np.any(abs_jumps.sum(axis=1) != 1):
                raise ValueError("every event must be a nearest-neighbor unit jump")

        calculated_end = start + jumps.sum(axis=0, dtype=np.int64)
        if not np.array_equal(calculated_end, end):
            raise ValueError(
                f"end is inconsistent with jumps: expected {calculated_end}, got {end}"
            )

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "event_times", times)
        object.__setattr__(self, "jumps", jumps)

    @property
    def d(self) -> int:
        return int(self.start.size)

    @property
    def n_events(self) -> int:
        return int(self.event_times.size)

    def position_at(self, tau: float) -> IntArray:
        """Return the right-continuous covering-space position at ``tau``."""
        if not np.isfinite(tau) or tau < 0.0 or tau > self.duration:
            raise ValueError("tau must lie in [0, duration]")
        count = int(np.searchsorted(self.event_times, tau, side="right"))
        if count == 0:
            return self.start.copy()
        return self.start + self.jumps[:count].sum(axis=0, dtype=np.int64)

    def positions_after_events(self) -> IntArray:
        """Return positions immediately after each event."""
        if self.n_events == 0:
            return np.empty((0, self.d), dtype=np.int64)
        return self.start + np.cumsum(self.jumps, axis=0, dtype=np.int64)

    def translated(self, displacement: Sequence[int] | IntArray) -> "ContinuousPath":
        shift = _as_int_vector(displacement, name="displacement")
        if shift.shape != self.start.shape:
            raise ValueError("displacement has the wrong dimension")
        return ContinuousPath(
            duration=self.duration,
            start=self.start + shift,
            event_times=self.event_times,
            jumps=self.jumps,
            end=self.end + shift,
        )


def sample_continuous_bridge_covering(
    a: Sequence[int] | IntArray,
    b: Sequence[int] | IntArray,
    duration: float,
    t: float,
    rng: np.random.Generator,
    *,
    tail_tol: float = 1e-14,
) -> ContinuousPath:
    r"""Sample an exact continuous-time free bridge on ``Z^d``.

    In each coordinate, ``N_+`` and ``N_-`` are independent Poisson variables
    of mean ``t*duration``, conditioned on ``N_+ - N_- = b-a``. Conditional on
    the counts, jump times are independent uniform order statistics.
    """
    aa = _as_int_vector(a, name="a")
    bb = _as_int_vector(b, name="b")
    if aa.shape != bb.shape:
        raise ValueError("a and b must have equal dimension")
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be finite and nonnegative")
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("t must be finite and nonnegative")
    if not (0.0 < tail_tol < 1.0):
        raise ValueError("tail_tol must lie in (0, 1)")

    delta = bb - aa
    if duration == 0.0 or t == 0.0:
        if np.any(delta != 0):
            raise ValueError("nonzero displacement has zero free weight")
        return ContinuousPath(
            duration=duration,
            start=aa,
            event_times=np.empty(0, dtype=np.float64),
            jumps=np.empty((0, aa.size), dtype=np.int64),
            end=bb,
        )

    lam = t * duration
    event_times: list[float] = []
    jump_vectors: list[IntArray] = []

    # Avoid exact endpoint events in finite-precision pseudorandom arithmetic.
    zero_plus = np.nextafter(0.0, 1.0)
    one_minus = np.nextafter(1.0, 0.0)

    for alpha, signed_delta in enumerate(delta.tolist()):
        D = abs(int(signed_delta))
        m = _sample_bessel_pair_count(D, lam, rng, tail_tol=tail_tol)
        if signed_delta >= 0:
            n_plus = m + int(signed_delta)
            n_minus = m
        else:
            n_plus = m
            n_minus = m - int(signed_delta)

        for sign, count in ((1, n_plus), (-1, n_minus)):
            if count == 0:
                continue
            uniforms = np.clip(rng.random(count), zero_plus, one_minus)
            times = duration * uniforms
            for event_time in times.tolist():
                jump = np.zeros(aa.size, dtype=np.int64)
                jump[alpha] = sign
                event_times.append(float(event_time))
                jump_vectors.append(jump)

    if not event_times:
        times_array = np.empty(0, dtype=np.float64)
        jumps_array = np.empty((0, aa.size), dtype=np.int64)
    else:
        order = np.argsort(np.asarray(event_times), kind="mergesort")
        times_array = np.asarray(event_times, dtype=np.float64)[order]
        jumps_array = np.asarray(jump_vectors, dtype=np.int64)[order]

    return ContinuousPath(
        duration=duration,
        start=aa,
        event_times=times_array,
        jumps=jumps_array,
        end=bb,
    )


def _centered_torus_displacement(delta: int, L: int) -> int:
    """Return a short representative of ``delta`` modulo ``L``."""
    out = int(delta) % L
    if out > L // 2:
        out -= L
    return out


def _displaced_winding_weights_1d(
    delta: int,
    L: int,
    duration: float,
    t: float,
    *,
    tail_tol: float,
    max_winding: int = 1_000_000,
) -> tuple[IntArray, FloatArray]:
    r"""Controlled-support winding weights for an open torus bridge.

    A torus displacement ``delta`` has covering displacements
    ``delta + L*w`` with weights ``I_|delta+L*w|(2*t*duration)``.  ``ive``
    removes the common exponential scale, which cancels in every winding and
    matching normalization below.
    """
    if L < 1:
        raise ValueError("L must be positive")
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be finite and nonnegative")
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("t must be finite and nonnegative")
    if not (0.0 < tail_tol < 1.0):
        raise ValueError("tail_tol must lie in (0, 1)")

    delta = _centered_torus_displacement(delta, L)
    z = 2.0 * t * duration
    if z == 0.0:
        if delta != 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        return np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64)

    w_max = max(4, int(ceil(8.0 * sqrt(max(z, 1.0)) / L)) + 4)
    log_tol = log(tail_tol)
    while True:
        if w_max > max_winding:
            raise RuntimeError("Displaced winding support exceeded max_winding")

        windings = np.arange(-w_max, w_max + 1, dtype=np.int64)
        orders = np.abs(delta + L * windings)
        weights = np.asarray(ive(orders, z), dtype=np.float64)
        weights = np.maximum(weights, 0.0)
        included = float(weights.sum())
        if included <= 0.0 or not np.isfinite(included):
            raise FloatingPointError("Failed to evaluate displaced winding weights")

        tail_bound = 0.0
        tails_controlled = True
        for sign in (-1, 1):
            first_w = sign * (w_max + 1)
            second_w = sign * (w_max + 2)
            first = max(0.0, float(ive(abs(delta + L * first_w), z)))
            second = max(0.0, float(ive(abs(delta + L * second_w), z)))
            if first == 0.0:
                continue
            ratio = second / first
            if ratio >= 1.0:
                tails_controlled = False
                break
            tail_bound += first / (1.0 - ratio)

        if tails_controlled and (
            tail_bound == 0.0 or log(tail_bound) - log(included) <= log_tol
        ):
            return windings, weights
        w_max = max(w_max + 8, int(1.5 * w_max) + 1)


def _log_torus_kernel_scaled(
    a: Sequence[int] | IntArray,
    b: Sequence[int] | IntArray,
    duration: float,
    L: int,
    t: float,
    *,
    tail_tol: float,
) -> float:
    r"""Log of the exponentially scaled finite-torus free kernel."""
    aa = _as_int_vector(a, name="a")
    bb = _as_int_vector(b, name="b")
    if aa.shape != bb.shape:
        raise ValueError("a and b must have equal dimension")

    total = 0.0
    for left, right in zip(aa.tolist(), bb.tolist()):
        _, weights = _displaced_winding_weights_1d(
            int(right) - int(left),
            L,
            duration,
            t,
            tail_tol=tail_tol,
        )
        value = float(weights.sum())
        if value <= 0.0:
            return float("-inf")
        total += log(value)
    return float(total)


def sample_continuous_bridge_torus(
    a: Sequence[int] | IntArray,
    b: Sequence[int] | IntArray,
    duration: float,
    L: int,
    t: float,
    rng: np.random.Generator,
    *,
    tail_tol: float = 1e-14,
) -> ContinuousPath:
    r"""Sample an exact continuous-time free bridge between torus sites.

    A covering endpoint is first sampled from the displaced winding law, then
    the existing covering-space Lévy bridge is used unchanged.  ``a`` may be
    any covering representative; only ``b modulo L`` is prescribed.
    """
    aa = _as_int_vector(a, name="a")
    bb = _as_int_vector(b, name="b")
    if aa.shape != bb.shape:
        raise ValueError("a and b must have equal dimension")
    if L < 1:
        raise ValueError("L must be positive")

    covering_end = np.empty_like(aa)
    for alpha, (left, right) in enumerate(zip(aa.tolist(), bb.tolist())):
        delta = _centered_torus_displacement(
            int(right) - (int(left) % L), L
        )
        windings, weights = _displaced_winding_weights_1d(
            delta,
            L,
            duration,
            t,
            tail_tol=tail_tol,
        )
        if weights.size == 0:
            raise ValueError("The requested torus bridge has zero free weight")
        winding = int(rng.choice(windings, p=weights / weights.sum()))
        covering_end[alpha] = int(left) + delta + L * winding

    return sample_continuous_bridge_covering(
        aa,
        covering_end,
        duration,
        t,
        rng,
        tail_tol=tail_tol,
    )


def split_continuous_path(
    path: ContinuousPath,
    cut_times: Iterable[float],
) -> list[ContinuousPath]:
    """Split a path at sorted internal cut times."""
    cuts = np.asarray(list(cut_times), dtype=np.float64)
    if cuts.ndim != 1:
        raise ValueError("cut_times must be one-dimensional")
    if cuts.size and (
        np.any(cuts <= 0.0)
        or np.any(cuts >= path.duration)
        or np.any(cuts[1:] <= cuts[:-1])
    ):
        raise ValueError("cut_times must be strictly increasing and internal")

    boundaries = np.concatenate(([0.0], cuts, [path.duration]))
    segments: list[ContinuousPath] = []

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        start = path.position_at(float(left))
        end = path.position_at(float(right))
        # Events at the left boundary have already been included in start.
        mask = (path.event_times > left) & (path.event_times <= right)
        local_times = path.event_times[mask] - left
        local_jumps = path.jumps[mask]
        segments.append(
            ContinuousPath(
                duration=float(right - left),
                start=start,
                event_times=local_times,
                jumps=local_jumps,
                end=end,
            )
        )

    return segments


def resample_path_interval(
    path: ContinuousPath,
    tau0: float,
    tau1: float,
    t: float,
    rng: np.random.Generator,
    *,
    tail_tol: float = 1e-14,
) -> ContinuousPath:
    """Replace ``(tau0, tau1]`` by an exact free bridge with fixed endpoints."""
    if not np.isfinite(tau0) or not np.isfinite(tau1):
        raise ValueError("interval endpoints must be finite")
    if tau0 < 0.0 or tau1 > path.duration or tau1 <= tau0:
        raise ValueError("require 0 <= tau0 < tau1 <= path.duration")

    left_position = path.position_at(tau0)
    right_position = path.position_at(tau1)
    proposal = sample_continuous_bridge_covering(
        left_position,
        right_position,
        tau1 - tau0,
        t,
        rng,
        tail_tol=tail_tol,
    )

    keep_before = path.event_times <= tau0
    keep_after = path.event_times > tau1
    new_times = np.concatenate(
        (
            path.event_times[keep_before],
            tau0 + proposal.event_times,
            path.event_times[keep_after],
        )
    )
    new_jumps = np.concatenate(
        (
            path.jumps[keep_before],
            proposal.jumps,
            path.jumps[keep_after],
        ),
        axis=0,
    )

    return ContinuousPath(
        duration=path.duration,
        start=path.start,
        event_times=new_times,
        jumps=new_jumps,
        end=path.end,
    )


@dataclass
class ContinuousConfiguration:
    """Canonical bosonic world lines on an ``L^d`` torus."""

    L: int
    d: int
    N: int
    beta: float
    t: float
    cycles: list[tuple[int, ...]]
    permutation: IntArray
    worldlines: list[ContinuousPath]
    log_Z0_N: float

    def validate(self) -> None:
        if self.N < 0:
            raise ValueError("N must be nonnegative")
        if self.L < 1 or self.d < 1:
            raise ValueError("L and d must be positive")
        if len(self.worldlines) != self.N:
            raise ValueError("worldlines must contain N paths")
        permutation = np.asarray(self.permutation, dtype=np.int64)
        if permutation.shape != (self.N,):
            raise ValueError("permutation has the wrong shape")
        if sorted(permutation.tolist()) != list(range(self.N)):
            raise ValueError("permutation is not a permutation of particle labels")

        flattened = [label for cycle in self.cycles for label in cycle]
        if sorted(flattened) != list(range(self.N)):
            raise ValueError("cycles do not partition all labels")
        for cycle in self.cycles:
            if not cycle:
                raise ValueError("empty cycle")
            for j, label in enumerate(cycle):
                if permutation[label] != cycle[(j + 1) % len(cycle)]:
                    raise ValueError("cycle list and permutation disagree")

        for particle, path in enumerate(self.worldlines):
            if path.d != self.d or not np.isclose(path.duration, self.beta):
                raise ValueError("worldline dimension or duration mismatch")
            next_particle = int(permutation[particle])
            if np.any(np.mod(path.end - self.worldlines[next_particle].start, self.L) != 0):
                raise ValueError("worldline endpoint does not match permutation target")

        self.permutation = permutation

    def copy(self) -> "ContinuousConfiguration":
        return ContinuousConfiguration(
            L=self.L,
            d=self.d,
            N=self.N,
            beta=self.beta,
            t=self.t,
            cycles=[tuple(cycle) for cycle in self.cycles],
            permutation=self.permutation.copy(),
            worldlines=list(self.worldlines),
            log_Z0_N=self.log_Z0_N,
        )

    @property
    def n_events(self) -> int:
        return int(sum(path.n_events for path in self.worldlines))

    @property
    def cycle_lengths(self) -> tuple[int, ...]:
        return tuple(len(cycle) for cycle in self.cycles)

    def positions_at(self, tau: float) -> IntArray:
        return np.asarray(
            [np.mod(path.position_at(tau), self.L) for path in self.worldlines],
            dtype=np.int64,
        )

    def total_winding(self) -> IntArray:
        displacement = np.zeros(self.d, dtype=np.int64)
        for path in self.worldlines:
            displacement += path.end - path.start
        if np.any(displacement % self.L != 0):
            raise RuntimeError("total covering displacement is not an integer winding")
        return displacement // self.L


def _cycles_from_permutation(permutation: IntArray) -> list[tuple[int, ...]]:
    """Return a deterministic cycle decomposition of a permutation."""
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.ndim != 1 or sorted(perm.tolist()) != list(range(perm.size)):
        raise ValueError("permutation is invalid")

    seen = np.zeros(perm.size, dtype=bool)
    cycles: list[tuple[int, ...]] = []
    for root in range(perm.size):
        if seen[root]:
            continue
        cycle: list[int] = []
        current = root
        while not seen[current]:
            seen[current] = True
            cycle.append(current)
            current = int(perm[current])
        if current != root:
            raise ValueError("permutation traversal did not close at its root")
        cycles.append(tuple(cycle))
    return cycles


def _splice_path_interval(
    prefix_path: ContinuousPath,
    suffix_path: ContinuousPath,
    bridge: ContinuousPath,
    tau0: float,
    tau1: float,
    L: int,
) -> ContinuousPath:
    """Join a prefix, a torus bridge, and a possibly different suffix."""
    if not np.isclose(prefix_path.duration, suffix_path.duration):
        raise ValueError("prefix and suffix durations differ")
    if not np.isclose(bridge.duration, tau1 - tau0):
        raise ValueError("bridge duration does not match the splice window")

    left = prefix_path.position_at(tau0)
    right = suffix_path.position_at(tau1)
    if not np.array_equal(bridge.start, left):
        raise ValueError("bridge does not start at the prefix cut")
    if np.any(np.mod(bridge.end - right, L) != 0):
        raise ValueError("bridge does not end at the suffix torus site")

    keep_before = prefix_path.event_times <= tau0
    keep_after = suffix_path.event_times > tau1
    event_times = np.concatenate(
        (
            prefix_path.event_times[keep_before],
            tau0 + bridge.event_times,
            suffix_path.event_times[keep_after],
        )
    )
    jumps = np.concatenate(
        (
            prefix_path.jumps[keep_before],
            bridge.jumps,
            suffix_path.jumps[keep_after],
        ),
        axis=0,
    )
    end = bridge.end + (suffix_path.end - right)
    return ContinuousPath(
        duration=prefix_path.duration,
        start=prefix_path.start,
        event_times=event_times,
        jumps=jumps,
        end=end,
    )


def rotate_configuration_time_origin(
    state: ContinuousConfiguration,
    shift: float,
) -> ContinuousConfiguration:
    r"""Rotate every closed permutation cycle by ``shift`` in imaginary time."""
    if not np.isfinite(shift) or shift < 0.0 or shift >= state.beta:
        raise ValueError("shift must lie in [0, beta)")
    if shift == 0.0 or state.N == 0:
        return state.copy()

    beta = state.beta
    new_paths: list[ContinuousPath] = []
    for particle, path in enumerate(state.worldlines):
        successor = state.worldlines[int(state.permutation[particle])]
        # Put a jump exactly on the new seam at time zero.  Using the left
        # limit for ``start`` and retaining the jump preserves both the event
        # count and right-continuous positions for explicitly requested shifts
        # that coincide with an event time.
        before_shift = int(np.searchsorted(path.event_times, shift, side="left"))
        start = path.start + path.jumps[:before_shift].sum(
            axis=0, dtype=np.int64
        )

        suffix_mask = path.event_times >= shift
        prefix_mask = successor.event_times < shift
        event_times = np.concatenate(
            (
                path.event_times[suffix_mask] - shift,
                (beta - shift) + successor.event_times[prefix_mask],
            )
        )
        jumps = np.concatenate(
            (
                path.jumps[suffix_mask],
                successor.jumps[prefix_mask],
            ),
            axis=0,
        )
        translation = path.end - successor.start
        if np.any(translation % state.L != 0):
            raise RuntimeError("invalid path connectivity during time rotation")
        successor_before_shift = int(
            np.searchsorted(successor.event_times, shift, side="left")
        )
        end = successor.start + successor.jumps[:successor_before_shift].sum(
            axis=0, dtype=np.int64
        ) + translation
        new_paths.append(
            ContinuousPath(
                duration=beta,
                start=start,
                event_times=event_times,
                jumps=jumps,
                end=end,
            )
        )

    rotated = ContinuousConfiguration(
        L=state.L,
        d=state.d,
        N=state.N,
        beta=state.beta,
        t=state.t,
        cycles=[tuple(cycle) for cycle in state.cycles],
        permutation=state.permutation.copy(),
        worldlines=new_paths,
        log_Z0_N=state.log_Z0_N,
    )
    rotated.validate()
    return rotated


@dataclass
class _SiteTimeline:
    """Dynamic exact occupancy timeline for one torus site."""

    initial: int = 0
    deltas: dict[float, int] = field(default_factory=dict)
    _dirty: bool = field(default=True, init=False)
    _times: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), init=False
    )
    _areas_before: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), init=False
    )
    _occupancies_after: IntArray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64), init=False
    )

    def adjust_initial(self, delta: int) -> None:
        self.initial += int(delta)
        if self.initial < 0:
            raise RuntimeError("negative site occupancy")
        self._dirty = True

    def adjust_event(self, event_time: float, delta: int) -> None:
        key = float(event_time)
        updated = self.deltas.get(key, 0) + int(delta)
        if updated:
            self.deltas[key] = updated
        else:
            self.deltas.pop(key, None)
        self._dirty = True

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        if not self.deltas:
            self._times = np.empty(0, dtype=np.float64)
            self._areas_before = np.empty(0, dtype=np.float64)
            self._occupancies_after = np.empty(0, dtype=np.int64)
            self._dirty = False
            return

        times = np.asarray(sorted(self.deltas), dtype=np.float64)
        areas_before = np.empty(times.size, dtype=np.float64)
        occupancies_after = np.empty(times.size, dtype=np.int64)
        occupancy = int(self.initial)
        area = 0.0
        previous = 0.0
        for index, event_time in enumerate(times.tolist()):
            area += (event_time - previous) * occupancy
            areas_before[index] = area
            occupancy += self.deltas[event_time]
            if occupancy < 0:
                raise RuntimeError("negative occupancy in site timeline")
            occupancies_after[index] = occupancy
            previous = event_time
        self._times = times
        self._areas_before = areas_before
        self._occupancies_after = occupancies_after
        self._dirty = False

    def integral_to(self, tau: float) -> float:
        self._rebuild()
        if self._times.size == 0:
            return float(self.initial * tau)
        index = int(np.searchsorted(self._times, tau, side="right")) - 1
        if index < 0:
            return float(self.initial * tau)
        return float(
            self._areas_before[index]
            + self._occupancies_after[index] * (tau - self._times[index])
        )

    def integral(self, tau0: float, tau1: float) -> float:
        return self.integral_to(tau1) - self.integral_to(tau0)

    def pair_integral(self, beta: float) -> float:
        self._rebuild()
        occupancy = int(self.initial)
        previous = 0.0
        total = 0.0
        for event_time in self._times.tolist():
            total += (event_time - previous) * occupancy * (occupancy - 1) / 2
            occupancy += self.deltas[event_time]
            previous = event_time
        total += (beta - previous) * occupancy * (occupancy - 1) / 2
        return float(total)


@dataclass
class _OccupancyIndex:
    """Space-time occupancy ledger for exact local action differences."""

    L: int
    d: int
    beta: float
    timelines: dict[tuple[int, ...], _SiteTimeline] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: ContinuousConfiguration) -> "_OccupancyIndex":
        out = cls(L=state.L, d=state.d, beta=state.beta)
        for path in state.worldlines:
            out.adjust_path(path, +1)
        return out

    def _site_key(self, position: IntArray) -> tuple[int, ...]:
        return tuple(int(x) for x in np.mod(position, self.L))

    def _timeline(self, key: tuple[int, ...]) -> _SiteTimeline:
        timeline = self.timelines.get(key)
        if timeline is None:
            timeline = _SiteTimeline()
            self.timelines[key] = timeline
        return timeline

    def adjust_path(self, path: ContinuousPath, sign: int) -> None:
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        position = np.mod(path.start, self.L).astype(np.int64)
        self._timeline(self._site_key(position)).adjust_initial(sign)
        for event_time, jump in zip(path.event_times.tolist(), path.jumps):
            old_key = self._site_key(position)
            position = np.mod(position + jump, self.L)
            new_key = self._site_key(position)
            if old_key == new_key:
                continue
            self._timeline(old_key).adjust_event(event_time, -sign)
            self._timeline(new_key).adjust_event(event_time, +sign)

    def integrate_path_occupancy(self, path: ContinuousPath) -> float:
        position = np.mod(path.start, self.L).astype(np.int64)
        previous = 0.0
        total = 0.0
        for event_time, jump in zip(path.event_times.tolist(), path.jumps):
            total += self._timeline(self._site_key(position)).integral(
                previous, event_time
            )
            position = np.mod(position + jump, self.L)
            previous = event_time
        total += self._timeline(self._site_key(position)).integral(
            previous, path.duration
        )
        return float(total)

    def replace_paths(
        self,
        old_paths: Sequence[ContinuousPath],
        new_paths: Sequence[ContinuousPath],
        current_overlap: float,
    ) -> float:
        removed = 0.0
        for path in old_paths:
            removed += self.integrate_path_occupancy(path) - path.duration
            self.adjust_path(path, -1)

        added = 0.0
        for path in new_paths:
            added += self.integrate_path_occupancy(path)
            self.adjust_path(path, +1)
        return float(current_overlap - removed + added)

    def rollback_replacement(
        self,
        old_paths: Sequence[ContinuousPath],
        new_paths: Sequence[ContinuousPath],
    ) -> None:
        for path in reversed(new_paths):
            self.adjust_path(path, -1)
        for path in old_paths:
            self.adjust_path(path, +1)

    def pair_overlap(self) -> float:
        return float(
            sum(timeline.pair_integral(self.beta) for timeline in self.timelines.values())
        )


def _paths_for_cycle(
    labels: tuple[int, ...],
    beta: float,
    L: int,
    d: int,
    t: float,
    rng: np.random.Generator,
    *,
    tail_tol: float,
) -> dict[int, ContinuousPath]:
    ell = len(labels)
    duration = ell * beta
    base = rng.integers(0, L, size=d, dtype=np.int64)
    winding = np.asarray(
        [
            sample_winding_1d(L, duration, t, rng, tail_tol=tail_tol)
            for _ in range(d)
        ],
        dtype=np.int64,
    )
    long_path = sample_continuous_bridge_covering(
        base,
        base + L * winding,
        duration,
        t,
        rng,
        tail_tol=tail_tol,
    )
    cuts = [j * beta for j in range(1, ell)]
    pieces = split_continuous_path(long_path, cuts)
    return {label: piece for label, piece in zip(labels, pieces)}


def sample_ideal_continuous_configuration(
    N: int,
    beta: float,
    L: int,
    d: int,
    t: float,
    rng: np.random.Generator | None = None,
    *,
    tail_tol: float = 1e-14,
) -> ContinuousConfiguration:
    """Sample an exact canonical ideal-boson continuous-time configuration."""
    _validate_model(beta, L, d, t, 0.0)
    if N < 0:
        raise ValueError("N must be nonnegative")
    if rng is None:
        rng = np.random.default_rng()

    log_z, log_Z = canonical_log_partition(N, beta, L, d, t)
    cycles = sample_cycle_labels(N, log_z, log_Z, rng)
    permutation = np.empty(N, dtype=np.int64)
    worldlines: list[ContinuousPath | None] = [None] * N

    for cycle in cycles:
        for j, label in enumerate(cycle):
            permutation[label] = cycle[(j + 1) % len(cycle)]
        proposed = _paths_for_cycle(
            cycle, beta, L, d, t, rng, tail_tol=tail_tol
        )
        for label, path in proposed.items():
            worldlines[label] = path

    state = ContinuousConfiguration(
        L=L,
        d=d,
        N=N,
        beta=beta,
        t=t,
        cycles=[tuple(cycle) for cycle in cycles],
        permutation=permutation,
        worldlines=[path for path in worldlines if path is not None],
        log_Z0_N=float(log_Z[N]),
    )
    state.validate()
    return state


def pair_overlap_time(state: ContinuousConfiguration) -> float:
    r"""Return ``integral d tau sum_x n_x(n_x-1)/2`` exactly."""
    if state.N < 2:
        return 0.0

    positions = [np.mod(path.start, state.L).astype(np.int64) for path in state.worldlines]
    counts: dict[tuple[int, ...], int] = {}
    for position in positions:
        key = tuple(int(x) for x in position)
        counts[key] = counts.get(key, 0) + 1

    pair_count = sum(value * (value - 1) // 2 for value in counts.values())
    events: list[tuple[float, int, IntArray]] = []
    for particle, path in enumerate(state.worldlines):
        for event_time, jump in zip(path.event_times.tolist(), path.jumps):
            events.append((float(event_time), particle, jump))
    events.sort(key=lambda item: item[0])

    overlap = 0.0
    previous_time = 0.0
    index = 0
    while index < len(events):
        event_time = events[index][0]
        overlap += (event_time - previous_time) * pair_count

        # All simultaneous events have zero-duration ordering ambiguity. Apply
        # them as a group after accumulating the preceding interval.
        group_end = index + 1
        while group_end < len(events) and events[group_end][0] == event_time:
            group_end += 1

        for _, particle, jump in events[index:group_end]:
            old_key = tuple(int(x) for x in positions[particle])
            old_occupancy = counts[old_key]
            pair_count -= old_occupancy - 1
            if old_occupancy == 1:
                del counts[old_key]
            else:
                counts[old_key] = old_occupancy - 1

            positions[particle] = np.mod(positions[particle] + jump, state.L)
            new_key = tuple(int(x) for x in positions[particle])
            new_occupancy = counts.get(new_key, 0)
            pair_count += new_occupancy
            counts[new_key] = new_occupancy + 1

        previous_time = event_time
        index = group_end

    overlap += (state.beta - previous_time) * pair_count
    return float(overlap)


def interaction_action(
    state: ContinuousConfiguration,
    U: float,
    *,
    mu: float = 0.0,
) -> float:
    """Exact diagonal action; the chemical-potential term is constant at fixed N."""
    if not np.isfinite(U) or not np.isfinite(mu):
        raise ValueError("U and mu must be finite")
    return float(U * pair_overlap_time(state) - mu * state.N * state.beta)


def kinetic_energy_estimator(state: ContinuousConfiguration) -> float:
    r"""Continuous-time expansion estimator ``<H0> = -<K>/beta``."""
    return -state.n_events / state.beta


def interaction_energy_estimator(state: ContinuousConfiguration, U: float) -> float:
    return U * pair_overlap_time(state) / state.beta


def total_energy_estimator(state: ContinuousConfiguration, U: float) -> float:
    return kinetic_energy_estimator(state) + interaction_energy_estimator(state, U)


def double_occupancy_per_site(state: ContinuousConfiguration) -> float:
    r"""Time-averaged ``sum_x n_x(n_x-1)/2`` divided by the number of sites."""
    volume = state.L ** state.d
    return pair_overlap_time(state) / (state.beta * volume)


def _metropolis_accept(delta_action: float, rng: np.random.Generator) -> bool:
    if delta_action <= 0.0:
        return True
    if delta_action >= 745.0:
        return False
    return bool(np.log(rng.random()) < -delta_action)


@dataclass
class MoveStatistics:
    attempts: int = 0
    accepts: int = 0
    topology_changes: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepts / self.attempts if self.attempts else float("nan")

    @property
    def topology_change_rate(self) -> float:
        return (
            self.topology_changes / self.attempts
            if self.attempts
            else float("nan")
        )


@dataclass
class InteractingLatticeLevySampler:
    """Exact canonical continuous-time Monte Carlo for finite ``U``.

    Ergodicity across permutation sectors is supplied by ``stitch_update`` or
    the legacy ``global_update``.  Stitching redraws a short closed slab with
    exact Lévy bridges and can split or merge cycles without opening a path.
    """

    N: int
    beta: float
    L: int
    d: int
    t: float
    U: float
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    tail_tol: float = 1e-14
    state: ContinuousConfiguration = field(init=False)
    pair_overlap: float = field(init=False)
    action: float = field(init=False)
    statistics: dict[str, MoveStatistics] = field(init=False)
    _occupancy_index: _OccupancyIndex = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_model(self.beta, self.L, self.d, self.t, self.U)
        if self.N < 0:
            raise ValueError("N must be nonnegative")
        self.state = sample_ideal_continuous_configuration(
            self.N,
            self.beta,
            self.L,
            self.d,
            self.t,
            self.rng,
            tail_tol=self.tail_tol,
        )
        self.pair_overlap = pair_overlap_time(self.state)
        self.action = self.U * self.pair_overlap
        self._occupancy_index = _OccupancyIndex.from_state(self.state)
        self.statistics = {
            "segment": MoveStatistics(),
            "cycle": MoveStatistics(),
            "stitch": MoveStatistics(),
            "time_shift": MoveStatistics(),
            "global": MoveStatistics(),
        }

    def _try_paths_replacement(
        self,
        replacements: dict[int, ContinuousPath],
        move_name: str,
    ) -> bool:
        stats = self.statistics[move_name]
        stats.attempts += 1
        labels = sorted(replacements)
        old_paths = [self.state.worldlines[label] for label in labels]
        new_paths = [replacements[label] for label in labels]
        new_overlap = self._occupancy_index.replace_paths(
            old_paths,
            new_paths,
            self.pair_overlap,
        )
        new_action = self.U * new_overlap
        accepted = _metropolis_accept(new_action - self.action, self.rng)
        if accepted:
            for label, path in replacements.items():
                self.state.worldlines[label] = path
            self.pair_overlap = new_overlap
            self.action = new_action
            stats.accepts += 1
        else:
            self._occupancy_index.rollback_replacement(old_paths, new_paths)
        return accepted

    def segment_update(
        self,
        particle: int | None = None,
        tau0: float | None = None,
        tau1: float | None = None,
        *,
        fraction: float = 0.25,
    ) -> bool:
        """Redraw one fixed-endpoint segment from the exact free bridge."""
        if self.N == 0:
            return True
        if particle is None:
            particle = int(self.rng.integers(self.N))
        if not 0 <= particle < self.N:
            raise ValueError("particle label out of range")

        if tau0 is None and tau1 is None:
            if not (0.0 < fraction <= 1.0):
                raise ValueError("fraction must lie in (0, 1]")
            duration = fraction * self.beta
            tau0 = 0.0 if duration == self.beta else float(
                self.rng.uniform(0.0, self.beta - duration)
            )
            tau1 = tau0 + duration
        elif tau0 is None or tau1 is None:
            raise ValueError("provide both tau0 and tau1, or neither")

        proposal = resample_path_interval(
            self.state.worldlines[particle],
            float(tau0),
            float(tau1),
            self.t,
            self.rng,
            tail_tol=self.tail_tol,
        )
        return self._try_paths_replacement({particle: proposal}, "segment")

    def whole_worldline_update(self, particle: int | None = None) -> bool:
        return self.segment_update(particle, 0.0, self.beta)

    def cycle_update(self, cycle_index: int | None = None) -> bool:
        """Redraw one complete cycle, including its base point and winding."""
        if not self.state.cycles:
            return True
        if cycle_index is None:
            cycle_index = int(self.rng.integers(len(self.state.cycles)))
        if not 0 <= cycle_index < len(self.state.cycles):
            raise ValueError("cycle_index out of range")
        cycle = self.state.cycles[cycle_index]
        replacements = _paths_for_cycle(
            cycle,
            self.beta,
            self.L,
            self.d,
            self.t,
            self.rng,
            tail_tol=self.tail_tol,
        )
        return self._try_paths_replacement(replacements, "cycle")

    def _sample_stitch_pair_proposal(
        self,
        particle: int,
        partner: int,
        tau0: float,
        tau1: float,
    ) -> tuple[dict[int, ContinuousPath], IntArray, bool]:
        """Sample the exact ideal conditional for a two-strand slab."""
        if particle == partner:
            raise ValueError("stitch particles must be distinct")
        old_i = self.state.worldlines[particle]
        old_j = self.state.worldlines[partner]
        duration = tau1 - tau0

        left_i = old_i.position_at(tau0)
        left_j = old_j.position_at(tau0)
        right_i = old_i.position_at(tau1)
        right_j = old_j.position_at(tau1)

        log_identity = _log_torus_kernel_scaled(
            left_i,
            right_i,
            duration,
            self.L,
            self.t,
            tail_tol=self.tail_tol,
        ) + _log_torus_kernel_scaled(
            left_j,
            right_j,
            duration,
            self.L,
            self.t,
            tail_tol=self.tail_tol,
        )
        log_exchange = _log_torus_kernel_scaled(
            left_i,
            right_j,
            duration,
            self.L,
            self.t,
            tail_tol=self.tail_tol,
        ) + _log_torus_kernel_scaled(
            left_j,
            right_i,
            duration,
            self.L,
            self.t,
            tail_tol=self.tail_tol,
        )

        if not np.isfinite(log_identity) and not np.isfinite(log_exchange):
            raise FloatingPointError("Both stitch matchings have zero free weight")
        if not np.isfinite(log_exchange):
            exchanged = False
        elif not np.isfinite(log_identity):
            exchanged = True
        else:
            log_normalizer = float(np.logaddexp(log_identity, log_exchange))
            exchanged = bool(
                np.log(self.rng.random()) < log_exchange - log_normalizer
            )

        suffix_i = old_j if exchanged else old_i
        suffix_j = old_i if exchanged else old_j
        target_i = right_j if exchanged else right_i
        target_j = right_i if exchanged else right_j

        bridge_i = sample_continuous_bridge_torus(
            left_i,
            target_i,
            duration,
            self.L,
            self.t,
            self.rng,
            tail_tol=self.tail_tol,
        )
        bridge_j = sample_continuous_bridge_torus(
            left_j,
            target_j,
            duration,
            self.L,
            self.t,
            self.rng,
            tail_tol=self.tail_tol,
        )
        replacements = {
            particle: _splice_path_interval(
                old_i, suffix_i, bridge_i, tau0, tau1, self.L
            ),
            partner: _splice_path_interval(
                old_j, suffix_j, bridge_j, tau0, tau1, self.L
            ),
        }

        permutation = self.state.permutation.copy()
        if exchanged:
            permutation[particle], permutation[partner] = (
                permutation[partner],
                permutation[particle],
            )
        return replacements, permutation, exchanged

    def _try_stitch_pair(
        self,
        particle: int,
        partner: int,
        tau0: float,
        tau1: float,
    ) -> bool:
        stats = self.statistics["stitch"]
        stats.attempts += 1
        replacements, permutation, exchanged = self._sample_stitch_pair_proposal(
            particle,
            partner,
            tau0,
            tau1,
        )
        labels = sorted(replacements)
        old_paths = [self.state.worldlines[label] for label in labels]
        new_paths = [replacements[label] for label in labels]
        new_overlap = self._occupancy_index.replace_paths(
            old_paths,
            new_paths,
            self.pair_overlap,
        )
        new_action = self.U * new_overlap
        accepted = _metropolis_accept(new_action - self.action, self.rng)
        if accepted:
            for label, path in replacements.items():
                self.state.worldlines[label] = path
            self.state.permutation = permutation
            self.state.cycles = _cycles_from_permutation(permutation)
            self.pair_overlap = new_overlap
            self.action = new_action
            stats.accepts += 1
            if exchanged:
                stats.topology_changes += 1
        else:
            self._occupancy_index.rollback_replacement(old_paths, new_paths)
        return accepted

    def _select_stitch_partner(
        self,
        particle: int,
        left_positions: IntArray,
        buckets: dict[tuple[int, ...], list[int]],
        *,
        locality_radius: int,
        global_partner_probability: float,
    ) -> int:
        if locality_radius < 0:
            raise ValueError("locality_radius must be nonnegative")
        if not (0.0 <= global_partner_probability <= 1.0):
            raise ValueError("global_partner_probability must lie in [0, 1]")

        use_global = bool(self.rng.random() < global_partner_probability)
        local_candidates: list[int] = []
        if not use_global:
            center = left_positions[particle]
            neighbor_sites: set[tuple[int, ...]] = set()
            for offset in product(
                range(-locality_radius, locality_radius + 1),
                repeat=self.d,
            ):
                site = tuple(
                    int((center[axis] + offset[axis]) % self.L)
                    for axis in range(self.d)
                )
                neighbor_sites.add(site)
            for site in neighbor_sites:
                local_candidates.extend(buckets.get(site, ()))
            local_candidates = [
                candidate for candidate in local_candidates if candidate != particle
            ]

        if local_candidates:
            return int(self.rng.choice(np.asarray(local_candidates, dtype=np.int64)))

        draw = int(self.rng.integers(self.N - 1))
        return draw if draw < particle else draw + 1

    def stitch_update(
        self,
        particle: int | None = None,
        partner: int | None = None,
        tau0: float | None = None,
        tau1: float | None = None,
        *,
        fraction: float = 0.25,
        locality_radius: int = 1,
        global_partner_probability: float = 0.05,
    ) -> bool:
        r"""Closed Lévy seam-stitch update that can split or merge cycles.

        A short slab is cut from two labeled paths.  Their right suffixes are
        retained or exchanged according to the exact ideal-gas conditional
        matching probability, and the slab is regenerated with exact torus
        Lévy bridges.  No path is opened.  The only Metropolis factor is
        ``exp(-Delta S_U)`` and that action difference is local to the slab.
        """
        if self.N < 2:
            return True
        if particle is None:
            particle = int(self.rng.integers(self.N))
        if not 0 <= particle < self.N:
            raise ValueError("particle label out of range")

        if tau0 is None and tau1 is None:
            if not (0.0 < fraction <= 1.0):
                raise ValueError("fraction must lie in (0, 1]")
            duration = fraction * self.beta
            tau0 = 0.0 if duration == self.beta else float(
                self.rng.uniform(0.0, self.beta - duration)
            )
            tau1 = min(self.beta, tau0 + duration)
        elif tau0 is None or tau1 is None:
            raise ValueError("provide both tau0 and tau1, or neither")
        tau0 = float(tau0)
        tau1 = float(tau1)
        if tau0 < 0.0 or tau1 > self.beta or tau1 <= tau0:
            raise ValueError("require 0 <= tau0 < tau1 <= beta")

        if partner is None:
            left_positions = self.state.positions_at(tau0)
            buckets: dict[tuple[int, ...], list[int]] = {}
            for label, position in enumerate(left_positions):
                key = tuple(int(x) for x in position)
                buckets.setdefault(key, []).append(label)
            partner = self._select_stitch_partner(
                particle,
                left_positions,
                buckets,
                locality_radius=locality_radius,
                global_partner_probability=global_partner_probability,
            )
        if not 0 <= partner < self.N or partner == particle:
            raise ValueError("partner must be a distinct valid particle label")
        return self._try_stitch_pair(particle, partner, tau0, tau1)

    def stitch_sweep(
        self,
        updates: int | None = None,
        *,
        fraction: float = 0.25,
        tau0: float | None = None,
        locality_radius: int = 1,
        global_partner_probability: float = 0.05,
    ) -> None:
        """Amortized stitch moves at one seam with an ``O(N)`` bucket build."""
        if self.N < 2:
            return
        if updates is None:
            updates = self.N
        if updates < 0:
            raise ValueError("updates must be nonnegative")
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must lie in (0, 1]")
        duration = fraction * self.beta
        if tau0 is None:
            tau0 = 0.0 if duration == self.beta else float(
                self.rng.uniform(0.0, self.beta - duration)
            )
        tau0 = float(tau0)
        tau1 = min(self.beta, tau0 + duration)
        if tau0 < 0.0 or tau1 > self.beta or tau1 <= tau0:
            raise ValueError("stitch window lies outside [0, beta]")

        left_positions = self.state.positions_at(tau0)
        buckets: dict[tuple[int, ...], list[int]] = {}
        for label, position in enumerate(left_positions):
            key = tuple(int(x) for x in position)
            buckets.setdefault(key, []).append(label)

        for _ in range(updates):
            particle = int(self.rng.integers(self.N))
            partner = self._select_stitch_partner(
                particle,
                left_positions,
                buckets,
                locality_radius=locality_radius,
                global_partner_probability=global_partner_probability,
            )
            self._try_stitch_pair(particle, partner, tau0, tau1)

    def time_shift_update(self, shift: float | None = None) -> bool:
        """Rejection-free random rotation of the imaginary-time origin."""
        stats = self.statistics["time_shift"]
        stats.attempts += 1
        if shift is None:
            shift = float(self.rng.uniform(0.0, self.beta))
        self.state = rotate_configuration_time_origin(self.state, float(shift))
        self._occupancy_index = _OccupancyIndex.from_state(self.state)
        stats.accepts += 1
        return True

    def random_seam_stitch_sweep(
        self,
        updates: int | None = None,
        *,
        fraction: float = 0.35,
        locality_radius: int = 1,
        global_partner_probability: float = 0.05,
    ) -> None:
        r"""Strictly reversible random-seam stitch macro-kernel.

        Let ``A`` denote the rejection-free uniform imaginary-time rotation
        and ``B`` the fixed-seam random-scan stitch kernel.  Both are
        self-adjoint with respect to the target measure; therefore

            A B**updates A

        also satisfies detailed balance.  The two rotations expose a random
        physical slab while permitting the partner buckets for all ``B``
        attempts to be built only once.
        """
        self.time_shift_update()
        self.stitch_sweep(
            updates=updates,
            fraction=fraction,
            tau0=0.0,
            locality_radius=locality_radius,
            global_partner_probability=global_partner_probability,
        )
        self.time_shift_update()

    def global_update(self) -> bool:
        """Exact ideal-gas independence proposal, including new cycle structure."""
        stats = self.statistics["global"]
        stats.attempts += 1
        proposal = sample_ideal_continuous_configuration(
            self.N,
            self.beta,
            self.L,
            self.d,
            self.t,
            self.rng,
            tail_tol=self.tail_tol,
        )
        new_overlap = pair_overlap_time(proposal)
        new_action = self.U * new_overlap
        accepted = _metropolis_accept(new_action - self.action, self.rng)
        if accepted:
            self.state = proposal
            self.pair_overlap = new_overlap
            self.action = new_action
            self._occupancy_index = _OccupancyIndex.from_state(self.state)
            stats.accepts += 1
        return accepted

    def sweep(
        self,
        *,
        segment_updates: int | None = None,
        segment_fraction: float = 0.25,
        cycle_updates: int = 1,
        stitch_updates: int = 0,
        stitch_fraction: float = 0.25,
        stitch_locality_radius: int = 1,
        stitch_global_partner_probability: float = 0.05,
        time_shift_updates: int = 0,
        global_updates: int = 0,
    ) -> None:
        """Perform a configurable collection of exact Metropolis moves."""
        if segment_updates is None:
            segment_updates = self.N
        for _ in range(segment_updates):
            self.segment_update(fraction=segment_fraction)
        for _ in range(cycle_updates):
            self.cycle_update()
        if stitch_updates:
            self.stitch_sweep(
                updates=stitch_updates,
                fraction=stitch_fraction,
                locality_radius=stitch_locality_radius,
                global_partner_probability=stitch_global_partner_probability,
            )
        for _ in range(time_shift_updates):
            self.time_shift_update()
        for _ in range(global_updates):
            self.global_update()

    def observables(self) -> dict[str, object]:
        overlap = self.pair_overlap
        return {
            "action": self.action,
            "pair_overlap_time": overlap,
            "double_occupancy_per_site": overlap
            / (self.beta * self.L**self.d),
            "kinetic_energy": -self.state.n_events / self.beta,
            "interaction_energy": self.U * overlap / self.beta,
            "total_energy": (-self.state.n_events + self.U * overlap) / self.beta,
            "n_events": self.state.n_events,
            "winding": self.state.total_winding().copy(),
            "cycle_lengths": self.state.cycle_lengths,
        }

    def run(
        self,
        n_samples: int,
        *,
        burn_in: int = 0,
        thin: int = 1,
        segment_updates: int | None = None,
        segment_fraction: float = 0.25,
        cycle_updates: int = 1,
        stitch_updates: int = 0,
        stitch_fraction: float = 0.25,
        stitch_locality_radius: int = 1,
        stitch_global_partner_probability: float = 0.05,
        time_shift_updates: int = 0,
        global_updates: int = 0,
    ) -> list[dict[str, object]]:
        if n_samples < 0 or burn_in < 0 or thin < 1:
            raise ValueError("invalid run lengths")
        for _ in range(burn_in):
            self.sweep(
                segment_updates=segment_updates,
                segment_fraction=segment_fraction,
                cycle_updates=cycle_updates,
                stitch_updates=stitch_updates,
                stitch_fraction=stitch_fraction,
                stitch_locality_radius=stitch_locality_radius,
                stitch_global_partner_probability=stitch_global_partner_probability,
                time_shift_updates=time_shift_updates,
                global_updates=global_updates,
            )

        output: list[dict[str, object]] = []
        for _ in range(n_samples):
            for _ in range(thin):
                self.sweep(
                    segment_updates=segment_updates,
                    segment_fraction=segment_fraction,
                    cycle_updates=cycle_updates,
                    stitch_updates=stitch_updates,
                    stitch_fraction=stitch_fraction,
                    stitch_locality_radius=stitch_locality_radius,
                    stitch_global_partner_probability=stitch_global_partner_probability,
                    time_shift_updates=time_shift_updates,
                    global_updates=global_updates,
                )
            output.append(self.observables())
        return output


__all__ = [
    "ContinuousPath",
    "ContinuousConfiguration",
    "InteractingLatticeLevySampler",
    "MoveStatistics",
    "sample_continuous_bridge_covering",
    "sample_continuous_bridge_torus",
    "split_continuous_path",
    "resample_path_interval",
    "rotate_configuration_time_origin",
    "sample_ideal_continuous_configuration",
    "pair_overlap_time",
    "interaction_action",
    "kinetic_energy_estimator",
    "interaction_energy_estimator",
    "total_energy_estimator",
    "double_occupancy_per_site",
]
