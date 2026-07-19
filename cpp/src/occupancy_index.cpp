#include "occupancy_index.hpp"

#include "checked_math.hpp"
#include "qmc/free_numerics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace qmc::detail {

namespace {

std::int64_t checked_occupancy_add(const std::int64_t value, const std::int64_t delta) {
  if ((delta > 0 && value > std::numeric_limits<std::int64_t>::max() - delta) ||
      (delta < 0 && value < std::numeric_limits<std::int64_t>::min() - delta)) {
    throw std::overflow_error("site occupancy counter overflowed");
  }
  return value + delta;
}

} // namespace

void OccupancyIndex::SiteTimeline::adjust_initial(const std::int64_t delta) {
  initial = checked_occupancy_add(initial, delta);
  if (initial < 0) {
    throw std::logic_error("negative initial site occupancy");
  }
  dirty = true;
}

void OccupancyIndex::SiteTimeline::adjust_event(const double event_time, const std::int64_t delta) {
  if (!std::isfinite(event_time)) {
    throw std::invalid_argument("occupancy event time must be finite");
  }
  auto [entry, inserted] = deltas.try_emplace(event_time, 0);
  static_cast<void>(inserted);
  entry->second = checked_occupancy_add(entry->second, delta);
  if (entry->second == 0) {
    deltas.erase(entry);
  }
  dirty = true;
}

void OccupancyIndex::SiteTimeline::rebuild() {
  if (!dirty) {
    return;
  }
  times.clear();
  areas_before.clear();
  occupancies_after.clear();
  times.reserve(deltas.size());
  areas_before.reserve(deltas.size());
  occupancies_after.reserve(deltas.size());

  std::int64_t occupancy = initial;
  double area = 0.0;
  double previous = 0.0;
  for (const auto &[event_time, delta] : deltas) {
    area += (event_time - previous) * static_cast<double>(occupancy);
    times.push_back(event_time);
    areas_before.push_back(area);
    occupancy = checked_occupancy_add(occupancy, delta);
    if (occupancy < 0) {
      throw std::logic_error("negative occupancy in site timeline");
    }
    occupancies_after.push_back(occupancy);
    previous = event_time;
  }
  dirty = false;
}

double OccupancyIndex::SiteTimeline::integral_to(const double tau) {
  rebuild();
  const auto next = std::ranges::upper_bound(times, tau);
  if (next == times.begin()) {
    return static_cast<double>(initial) * tau;
  }
  const std::size_t index = static_cast<std::size_t>((next - times.begin()) - 1);
  return areas_before[index] +
         (static_cast<double>(occupancies_after[index]) * (tau - times[index]));
}

double OccupancyIndex::SiteTimeline::integral(const double tau0, const double tau1) {
  return integral_to(tau1) - integral_to(tau0);
}

double OccupancyIndex::SiteTimeline::pair_integral(const double beta) {
  rebuild();
  std::int64_t occupancy = initial;
  double previous = 0.0;
  double total = 0.0;
  for (const auto &[event_time, delta] : deltas) {
    const auto count = static_cast<double>(occupancy);
    total += (event_time - previous) * count * (count - 1.0) / 2.0;
    occupancy = checked_occupancy_add(occupancy, delta);
    previous = event_time;
  }
  const auto count = static_cast<double>(occupancy);
  total += (beta - previous) * count * (count - 1.0) / 2.0;
  return total;
}

OccupancyIndex::OccupancyIndex(const Model &model)
    : linear_size_(model.linear_size), dimension_(model.dimension), beta_(model.beta) {
  model.validate();
  if (model.beta <= 0.0) {
    throw std::invalid_argument("occupancy index requires positive beta");
  }
}

Site OccupancyIndex::site_key(const Site &position) const {
  if (position.size() != dimension_) {
    throw std::invalid_argument("occupancy position dimension mismatch");
  }
  Site key = position;
  for (Coord &coordinate : key) {
    coordinate = torus_mod(coordinate, linear_size_);
  }
  return key;
}

OccupancyIndex::SiteTimeline &OccupancyIndex::timeline(const Site &key) { return timelines_[key]; }

void OccupancyIndex::adjust_path(const ContinuousPath &path, const std::int64_t sign) {
  if (sign != -1 && sign != 1) {
    throw std::invalid_argument("occupancy path adjustment sign must be -1 or +1");
  }
  path.validate(dimension_);
  Site position = site_key(path.start);
  timeline(position).adjust_initial(sign);
  for (const JumpEvent &event : path.events) {
    const Site old_key = position;
    position[event.axis] =
        torus_mod(checked_add(position[event.axis], static_cast<Coord>(event.direction),
                              "occupancy path coordinate exceeds int64 range"),
                  linear_size_);
    if (old_key == position) {
      continue;
    }
    timeline(old_key).adjust_event(event.time, -sign);
    timeline(position).adjust_event(event.time, sign);
  }
}

double OccupancyIndex::integrate_path_occupancy(const ContinuousPath &path) {
  path.validate(dimension_);
  Site position = site_key(path.start);
  double previous = 0.0;
  double total = 0.0;
  for (const JumpEvent &event : path.events) {
    total += timeline(position).integral(previous, event.time);
    position[event.axis] =
        torus_mod(checked_add(position[event.axis], static_cast<Coord>(event.direction),
                              "occupancy path coordinate exceeds int64 range"),
                  linear_size_);
    previous = event.time;
  }
  total += timeline(position).integral(previous, path.duration);
  return total;
}

void OccupancyIndex::rebuild(const std::span<const ContinuousPath> paths) {
  timelines_.clear();
  for (const ContinuousPath &path : paths) {
    adjust_path(path, 1);
  }
}

double OccupancyIndex::replace_paths(const std::span<const ContinuousPath *const> old_paths,
                                     const std::span<const ContinuousPath *const> new_paths,
                                     const double current_overlap) {
  double removed = 0.0;
  for (const ContinuousPath *path : old_paths) {
    if (path == nullptr) {
      throw std::invalid_argument("old replacement path must not be null");
    }
    removed += integrate_path_occupancy(*path) - path->duration;
    adjust_path(*path, -1);
  }

  double added = 0.0;
  for (const ContinuousPath *path : new_paths) {
    if (path == nullptr) {
      throw std::invalid_argument("new replacement path must not be null");
    }
    added += integrate_path_occupancy(*path);
    adjust_path(*path, 1);
  }
  return current_overlap - removed + added;
}

void OccupancyIndex::rollback_replacement(const std::span<const ContinuousPath *const> old_paths,
                                          const std::span<const ContinuousPath *const> new_paths) {
  for (const ContinuousPath *path : std::views::reverse(new_paths)) {
    adjust_path(*path, -1);
  }
  for (const ContinuousPath *path : old_paths) {
    adjust_path(*path, 1);
  }
}

double OccupancyIndex::pair_overlap() {
  double total = 0.0;
  for (auto &[site, timeline_value] : timelines_) {
    static_cast<void>(site);
    total += timeline_value.pair_integral(beta_);
  }
  return total;
}

} // namespace qmc::detail
