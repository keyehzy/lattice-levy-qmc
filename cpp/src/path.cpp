#include "qmc/path.hpp"

#include "checked_math.hpp"
#include "path_cursor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qmc {
namespace {

void validate_duration(const double duration) {
  if (!std::isfinite(duration) || duration < 0.0) {
    throw std::invalid_argument("duration must be finite and nonnegative");
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

ContinuousPath::ContinuousPath(const double duration, Site start, Site end,
                               std::vector<JumpEvent> events)
    : duration_(duration), start_(std::move(start)), end_(std::move(end)),
      events_(std::move(events)) {
  validate(start_.size());
}

void ContinuousPath::validate(const std::size_t dimension) const {
  validate_duration(duration_);
  if (start_.size() != dimension || end_.size() != dimension) {
    throw std::invalid_argument("continuous path endpoint dimension mismatch");
  }

  Site calculated = start_;
  double previous_time = 0.0;
  bool first = true;
  for (const JumpEvent &event : events_) {
    if (!std::isfinite(event.time) || event.time < 0.0 || event.time > duration_) {
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
  if (calculated != end_) {
    throw std::invalid_argument("continuous path endpoint is inconsistent with its events");
  }
}

Site ContinuousPath::position_at(const double tau) const {
  if (!std::isfinite(tau) || tau < 0.0 || tau > duration_) {
    throw std::invalid_argument("tau must lie in [0, duration]");
  }

  const auto event_end = std::ranges::upper_bound(events_, tau, {}, &JumpEvent::time);
  Site position = start_;
  for (auto event = events_.begin(); event != event_end; ++event) {
    position[event->axis] = apply_direction(position[event->axis], event->direction);
  }
  return position;
}

std::vector<Site> ContinuousPath::positions_after_events() const {
  std::vector<Site> positions;
  positions.reserve(events_.size());
  Site position = start_;
  for (const JumpEvent &event : events_) {
    position[event.axis] = apply_direction(position[event.axis], event.direction);
    positions.push_back(position);
  }
  return positions;
}

ContinuousPath ContinuousPath::translated(const Site &displacement) const {
  if (displacement.size() != start_.size()) {
    throw std::invalid_argument("translation displacement has the wrong dimension");
  }
  Site translated_start(start_.size());
  Site translated_end(end_.size());
  for (std::size_t axis = 0; axis < start_.size(); ++axis) {
    translated_start[axis] = detail::checked_add(start_[axis], displacement[axis],
                                                 "translated path start exceeds int64 range");
    translated_end[axis] = detail::checked_add(end_[axis], displacement[axis],
                                               "translated path end exceeds int64 range");
  }
  return {duration_, std::move(translated_start), std::move(translated_end), events_};
}

ContinuousPath sample_continuous_bridge(const Site &a, const Site &b, const double duration,
                                        const FreePathKernels &kernels, Random &random) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("bridge endpoints must have equal dimensions");
  }
  validate_duration(duration);
  if (a.size() > static_cast<std::size_t>(std::numeric_limits<Axis>::max()) + std::size_t{1}) {
    throw std::invalid_argument("bridge dimension exceeds the Axis representation");
  }

  if (duration == 0.0 || kernels.hopping() == 0.0) {
    if (a != b) {
      throw std::invalid_argument("nonzero displacement has zero free bridge weight");
    }
    return {duration, a, b, {}};
  }

  const double lambda = kernels.hopping() * duration;
  if (!std::isfinite(lambda)) {
    throw std::overflow_error("hopping times continuous bridge duration overflowed");
  }

  std::vector<AxisCounts> counts;
  counts.reserve(a.size());
  std::size_t total_events = 0;
  for (std::size_t axis_index = 0; axis_index < a.size(); ++axis_index) {
    const auto [positive, abs_delta] = detail::displacement(a[axis_index], b[axis_index]);
    const std::uint64_t pairs = kernels.sample_bessel_pair_count(abs_delta, lambda, random);
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

  return {duration, a, b, std::move(events)};
}

ContinuousPath sample_continuous_bridge(const Site &a, const Site &b, const double duration,
                                        const double hopping, Random &random,
                                        const NumericalOptions &options) {
  return sample_continuous_bridge(a, b, duration, FreePathKernels(hopping, options), random);
}

ContinuousPath sample_continuous_bridge_torus(const Site &a, const Site &b, const double duration,
                                              const FreePathKernels &kernels, Random &random) {
  Site covering_end = kernels.sample_torus_covering_endpoint(a, b, duration, random);
  return sample_continuous_bridge(a, covering_end, duration, kernels, random);
}

ContinuousPath sample_continuous_bridge_torus(const Site &a, const Site &b, const double duration,
                                              const Coord linear_size, const double hopping,
                                              Random &random, const NumericalOptions &options) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("torus bridge endpoints must have equal dimensions");
  }
  const std::size_t dimension = std::max<std::size_t>(1, a.size());
  return sample_continuous_bridge_torus(
      a, b, duration, FreePathKernels(TorusLayout(linear_size, dimension), hopping, options),
      random);
}

double log_torus_kernel_scaled(const Site &a, const Site &b, const double duration,
                               const FreePathKernels &kernels) {
  return kernels.log_torus_kernel_scaled(a, b, duration);
}

double log_torus_kernel_scaled(const Site &a, const Site &b, const double duration,
                               const Coord linear_size, const double hopping,
                               const NumericalOptions &options) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("torus kernel endpoints must have equal dimensions");
  }
  const std::size_t dimension = std::max<std::size_t>(1, a.size());
  return FreePathKernels(TorusLayout(linear_size, dimension), hopping, options)
      .log_torus_kernel_scaled(a, b, duration);
}

std::vector<ContinuousPath> split_continuous_path(const ContinuousPath &path,
                                                  const std::span<const double> cut_times) {
  double previous = 0.0;
  for (const double cut : cut_times) {
    if (!std::isfinite(cut) || cut <= previous || cut >= path.duration()) {
      throw std::invalid_argument("cut times must be strictly increasing and internal");
    }
    previous = cut;
  }

  std::vector<ContinuousPath> pieces;
  pieces.reserve(cut_times.size() + 1);
  detail::PathCursor cursor(path);
  detail::PathCut left = cursor.cut(0.0);
  for (const double right_time : cut_times) {
    detail::PathCut right = cursor.cut(right_time);
    pieces.push_back(detail::materialize_path_slice(cursor.slice(left, right)));
    left = std::move(right);
  }
  const detail::PathCut right = cursor.cut(path.duration());
  pieces.push_back(detail::materialize_path_slice(cursor.slice(left, right)));
  return pieces;
}

ContinuousPath resample_path_interval(const ContinuousPath &path, const double tau0,
                                      const double tau1, const FreePathKernels &kernels,
                                      Random &random) {
  if (!std::isfinite(tau0) || !std::isfinite(tau1) || tau0 < 0.0 || tau1 > path.duration() ||
      tau1 <= tau0) {
    throw std::invalid_argument("require 0 <= tau0 < tau1 <= path duration");
  }

  detail::PathCursor cursor(path);
  const detail::PathCut left = cursor.cut(tau0);
  const detail::PathCut right = cursor.cut(tau1);
  const detail::PathSlice slice = cursor.slice(left, right);
  const ContinuousPath proposal =
      sample_continuous_bridge(slice.start, slice.end, tau1 - tau0, kernels, random);
  return detail::replace_path_slice(slice, proposal);
}

ContinuousPath resample_path_interval(const ContinuousPath &path, const double tau0,
                                      const double tau1, const double hopping, Random &random,
                                      const NumericalOptions &options) {
  return resample_path_interval(path, tau0, tau1, FreePathKernels(hopping, options), random);
}

} // namespace qmc
