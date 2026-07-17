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

There is no Trotter discretization.  ``tail_tol`` only controls the numerically
omitted tail in discrete inversion samplers for conditioned jump counts and
winding sectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

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

    @property
    def acceptance(self) -> float:
        return self.accepts / self.attempts if self.attempts else float("nan")


@dataclass
class InteractingLatticeLevySampler:
    """Exact canonical continuous-time Monte Carlo for finite ``U``.

    Ergodicity across permutation sectors is supplied by ``global_update``.
    Segment and cycle updates provide cheaper geometry and winding changes.
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
        self.statistics = {
            "segment": MoveStatistics(),
            "cycle": MoveStatistics(),
            "global": MoveStatistics(),
        }

    def _try_paths_replacement(
        self,
        replacements: dict[int, ContinuousPath],
        move_name: str,
    ) -> bool:
        stats = self.statistics[move_name]
        stats.attempts += 1
        old_paths = {label: self.state.worldlines[label] for label in replacements}
        for label, path in replacements.items():
            self.state.worldlines[label] = path

        new_overlap = pair_overlap_time(self.state)
        new_action = self.U * new_overlap
        accepted = _metropolis_accept(new_action - self.action, self.rng)
        if accepted:
            self.pair_overlap = new_overlap
            self.action = new_action
            stats.accepts += 1
        else:
            for label, path in old_paths.items():
                self.state.worldlines[label] = path
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
            stats.accepts += 1
        return accepted

    def sweep(
        self,
        *,
        segment_updates: int | None = None,
        segment_fraction: float = 0.25,
        cycle_updates: int = 1,
        global_updates: int = 0,
    ) -> None:
        """Perform a configurable collection of exact Metropolis moves."""
        if segment_updates is None:
            segment_updates = self.N
        for _ in range(segment_updates):
            self.segment_update(fraction=segment_fraction)
        for _ in range(cycle_updates):
            self.cycle_update()
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
        global_updates: int = 0,
    ) -> list[dict[str, object]]:
        if n_samples < 0 or burn_in < 0 or thin < 1:
            raise ValueError("invalid run lengths")
        for _ in range(burn_in):
            self.sweep(
                segment_updates=segment_updates,
                segment_fraction=segment_fraction,
                cycle_updates=cycle_updates,
                global_updates=global_updates,
            )

        output: list[dict[str, object]] = []
        for _ in range(n_samples):
            for _ in range(thin):
                self.sweep(
                    segment_updates=segment_updates,
                    segment_fraction=segment_fraction,
                    cycle_updates=cycle_updates,
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
    "split_continuous_path",
    "resample_path_interval",
    "sample_ideal_continuous_configuration",
    "pair_overlap_time",
    "interaction_action",
    "kinetic_energy_estimator",
    "interaction_energy_estimator",
    "total_energy_estimator",
    "double_occupancy_per_site",
]
