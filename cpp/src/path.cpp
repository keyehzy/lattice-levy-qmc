#include "qmc/path.hpp"

#include "checked_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmc {
namespace {

void validate_duration(const double duration) {
  if (!std::isfinite(duration) || duration < 0.0) {
    throw std::invalid_argument("duration must be finite and nonnegative");
  }
}

void validate_hopping(const double hopping) {
  if (!std::isfinite(hopping) || hopping < 0.0) {
    throw std::invalid_argument("hopping must be finite and nonnegative");
  }
}

Coord apply_direction(const Coord coordinate, const std::int8_t direction) {
  return detail::checked_add(coordinate, static_cast<Coord>(direction),
                             "continuous path coordinate exceeds int64 range");
}

double sample_internal_time(const double duration, Random &random) {
  double time = duration * random.uniform_open();
  if (time <= 0.0) {
    time = std::nextafter(0.0, duration);
  }
  if (time >= duration) {
    time = std::nextafter(duration, 0.0);
  }
  if (time <= 0.0 || time >= duration) {
    throw std::runtime_error("bridge duration has no representable internal event time");
  }
  return time;
}

struct AxisCounts {
  Axis axis;
  std::uint64_t plus;
  std::uint64_t minus;
};

} // namespace

void ContinuousPath::validate(const std::size_t dimension) const {
  validate_duration(duration);
  if (start.size() != dimension || end.size() != dimension) {
    throw std::invalid_argument("continuous path endpoint dimension mismatch");
  }

  Site calculated = start;
  double previous_time = 0.0;
  bool first = true;
  for (const JumpEvent &event : events) {
    if (!std::isfinite(event.time) || event.time < 0.0 || event.time > duration) {
      throw std::invalid_argument("continuous path event time lies outside [0, duration]");
    }
    if (!first && event.time < previous_time) {
      throw std::invalid_argument("continuous path events must be sorted by time");
    }
    if (static_cast<std::size_t>(event.axis) >= dimension) {
      throw std::invalid_argument("continuous path event axis is out of range");
    }
    if (event.direction != -1 && event.direction != 1) {
      throw std::invalid_argument("continuous path event direction must be -1 or +1");
    }
    calculated[event.axis] = apply_direction(calculated[event.axis], event.direction);
    previous_time = event.time;
    first = false;
  }
  if (calculated != end) {
    throw std::invalid_argument("continuous path endpoint is inconsistent with its events");
  }
}

Site ContinuousPath::position_at(const double tau) const {
  if (!std::isfinite(tau) || tau < 0.0 || tau > duration) {
    throw std::invalid_argument("tau must lie in [0, duration]");
  }
  validate(start.size());

  const auto event_end = std::ranges::upper_bound(events, tau, {}, &JumpEvent::time);
  Site position = start;
  for (auto event = events.begin(); event != event_end; ++event) {
    position[event->axis] = apply_direction(position[event->axis], event->direction);
  }
  return position;
}

std::vector<Site> ContinuousPath::positions_after_events() const {
  validate(start.size());
  std::vector<Site> positions;
  positions.reserve(events.size());
  Site position = start;
  for (const JumpEvent &event : events) {
    position[event.axis] = apply_direction(position[event.axis], event.direction);
    positions.push_back(position);
  }
  return positions;
}

ContinuousPath ContinuousPath::translated(const Site &displacement) const {
  validate(start.size());
  if (displacement.size() != start.size()) {
    throw std::invalid_argument("translation displacement has the wrong dimension");
  }
  Site translated_start(start.size());
  Site translated_end(end.size());
  for (std::size_t axis = 0; axis < start.size(); ++axis) {
    translated_start[axis] = detail::checked_add(start[axis], displacement[axis],
                                                 "translated path start exceeds int64 range");
    translated_end[axis] = detail::checked_add(end[axis], displacement[axis],
                                               "translated path end exceeds int64 range");
  }
  return ContinuousPath{
      .duration = duration,
      .start = std::move(translated_start),
      .end = std::move(translated_end),
      .events = events,
  };
}

ContinuousPath sample_continuous_bridge(const Site &a, const Site &b, const double duration,
                                        const double hopping, Random &random,
                                        const NumericalOptions &options) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("bridge endpoints must have equal dimensions");
  }
  validate_duration(duration);
  validate_hopping(hopping);
  options.validate();
  if (a.size() > static_cast<std::size_t>(std::numeric_limits<Axis>::max()) + std::size_t{1}) {
    throw std::invalid_argument("bridge dimension exceeds the Axis representation");
  }

  if (duration == 0.0 || hopping == 0.0) {
    if (a != b) {
      throw std::invalid_argument("nonzero displacement has zero free bridge weight");
    }
    return ContinuousPath{.duration = duration, .start = a, .end = b, .events = {}};
  }

  const double lambda = hopping * duration;
  if (!std::isfinite(lambda)) {
    throw std::overflow_error("hopping times continuous bridge duration overflowed");
  }

  std::vector<AxisCounts> counts;
  counts.reserve(a.size());
  std::size_t total_events = 0;
  for (std::size_t axis_index = 0; axis_index < a.size(); ++axis_index) {
    const auto [positive, abs_delta] = detail::displacement(a[axis_index], b[axis_index]);
    const std::uint64_t pairs = sample_bessel_pair_count(abs_delta, lambda, random, options);
    if (pairs > std::numeric_limits<std::uint64_t>::max() - abs_delta) {
      throw std::overflow_error("conditioned continuous jump count exceeds uint64 range");
    }
    const std::uint64_t larger = pairs + abs_delta;
    const std::uint64_t plus = positive ? larger : pairs;
    const std::uint64_t minus = positive ? pairs : larger;
    const auto plus_size = detail::checked_size(plus, "continuous jump count exceeds size_t");
    const auto minus_size = detail::checked_size(minus, "continuous jump count exceeds size_t");
    total_events =
        detail::checked_add_size(total_events, plus_size, "continuous event count exceeds size_t");
    total_events =
        detail::checked_add_size(total_events, minus_size, "continuous event count exceeds size_t");
    counts.push_back(AxisCounts{
        .axis = static_cast<Axis>(axis_index),
        .plus = plus,
        .minus = minus,
    });
  }

  std::vector<JumpEvent> events;
  if (total_events > events.max_size()) {
    throw std::length_error("continuous event count exceeds vector capacity");
  }
  events.reserve(total_events);
  for (const AxisCounts &axis_counts : counts) {
    for (std::uint64_t event = 0; event < axis_counts.plus; ++event) {
      events.push_back(JumpEvent{
          .time = sample_internal_time(duration, random),
          .axis = axis_counts.axis,
          .direction = 1,
      });
    }
    for (std::uint64_t event = 0; event < axis_counts.minus; ++event) {
      events.push_back(JumpEvent{
          .time = sample_internal_time(duration, random),
          .axis = axis_counts.axis,
          .direction = -1,
      });
    }
  }
  std::ranges::stable_sort(events, {}, &JumpEvent::time);

  ContinuousPath path{
      .duration = duration,
      .start = a,
      .end = b,
      .events = std::move(events),
  };
  path.validate(a.size());
  return path;
}

std::vector<ContinuousPath> split_continuous_path(const ContinuousPath &path,
                                                  const std::span<const double> cut_times) {
  path.validate(path.start.size());
  double previous = 0.0;
  for (const double cut : cut_times) {
    if (!std::isfinite(cut) || cut <= previous || cut >= path.duration) {
      throw std::invalid_argument("cut times must be strictly increasing and internal");
    }
    previous = cut;
  }

  std::vector<double> boundaries;
  boundaries.reserve(cut_times.size() + 2);
  boundaries.push_back(0.0);
  boundaries.insert(boundaries.end(), cut_times.begin(), cut_times.end());
  boundaries.push_back(path.duration);

  std::vector<ContinuousPath> pieces;
  pieces.reserve(boundaries.size() - 1);
  for (std::size_t piece_index = 0; piece_index + 1 < boundaries.size(); ++piece_index) {
    const double left = boundaries[piece_index];
    const double right = boundaries[piece_index + 1];
    std::vector<JumpEvent> local_events;
    for (const JumpEvent &event : path.events) {
      if (event.time > left && event.time <= right) {
        local_events.push_back(JumpEvent{
            .time = event.time - left,
            .axis = event.axis,
            .direction = event.direction,
        });
      }
    }
    ContinuousPath piece{
        .duration = right - left,
        .start = path.position_at(left),
        .end = path.position_at(right),
        .events = std::move(local_events),
    };
    piece.validate(path.start.size());
    pieces.push_back(std::move(piece));
  }
  return pieces;
}

ContinuousPath resample_path_interval(const ContinuousPath &path, const double tau0,
                                      const double tau1, const double hopping, Random &random,
                                      const NumericalOptions &options) {
  path.validate(path.start.size());
  if (!std::isfinite(tau0) || !std::isfinite(tau1) || tau0 < 0.0 || tau1 > path.duration ||
      tau1 <= tau0) {
    throw std::invalid_argument("require 0 <= tau0 < tau1 <= path duration");
  }

  const ContinuousPath proposal = sample_continuous_bridge(
      path.position_at(tau0), path.position_at(tau1), tau1 - tau0, hopping, random, options);
  std::vector<JumpEvent> events;
  events.reserve(path.events.size() + proposal.events.size());
  for (const JumpEvent &event : path.events) {
    if (event.time <= tau0) {
      events.push_back(event);
    }
  }
  for (const JumpEvent &event : proposal.events) {
    events.push_back(JumpEvent{
        .time = tau0 + event.time,
        .axis = event.axis,
        .direction = event.direction,
    });
  }
  for (const JumpEvent &event : path.events) {
    if (event.time > tau1) {
      events.push_back(event);
    }
  }

  ContinuousPath result{
      .duration = path.duration,
      .start = path.start,
      .end = path.end,
      .events = std::move(events),
  };
  result.validate(path.start.size());
  return result;
}

} // namespace qmc
