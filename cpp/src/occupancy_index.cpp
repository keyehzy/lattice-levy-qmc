#include "occupancy_index.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

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

bool OccupancyIndex::SiteTimeline::empty() const noexcept { return initial == 0 && deltas.empty(); }

OccupancyIndex::ReplacementTransaction::ReplacementTransaction(
    OccupancyIndex &owner, const double proposed_overlap, TimelineMap staged_timelines) noexcept
    : owner_(&owner), proposed_overlap_(proposed_overlap),
      staged_timelines_(std::move(staged_timelines)) {}

OccupancyIndex::ReplacementTransaction::ReplacementTransaction(
    ReplacementTransaction &&other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), proposed_overlap_(other.proposed_overlap_),
      staged_timelines_(std::move(other.staged_timelines_)) {}

void OccupancyIndex::ReplacementTransaction::commit() noexcept {
  static_assert(std::is_nothrow_move_constructible_v<TimelineMap>);
  static_assert(std::is_nothrow_swappable_v<SiteTimeline>);
  if (owner_ == nullptr) {
    return;
  }

  // Every key and value is already allocated. Existing values are swapped,
  // obsolete entries are erased, and new entries transfer their map nodes.
  // SiteLess and all involved destructors are non-throwing.
  while (!staged_timelines_.empty()) {
    auto staged = staged_timelines_.begin();
    auto accepted = owner_->timelines_.find(staged->first);
    if (staged->second.empty()) {
      if (accepted != owner_->timelines_.end()) {
        owner_->timelines_.erase(accepted);
      }
      staged_timelines_.erase(staged);
      continue;
    }
    if (accepted != owner_->timelines_.end()) {
      std::ranges::swap(accepted->second, staged->second);
      staged_timelines_.erase(staged);
      continue;
    }

    auto node = staged_timelines_.extract(staged);
    const auto inserted = owner_->timelines_.insert(std::move(node));
    if (!inserted.inserted) {
      std::terminate();
    }
  }
  owner_ = nullptr;
}

OccupancyIndex::OccupancyIndex(const Model &model)
    : layout_(model.linear_size, model.dimension), beta_(model.beta) {
  model.validate();
  if (model.beta <= 0.0) {
    throw std::invalid_argument("occupancy index requires positive beta");
  }
}

OccupancyIndex::SiteTimeline &OccupancyIndex::timeline(TimelineMap &timelines, const SiteId key) {
  return timelines[key];
}

void OccupancyIndex::stage_path_timelines(const ContinuousPath &path, TimelineMap &staged) const {
  path.validate(layout_.dimension());
  SiteId position = layout_.encode_covering(path.start);
  const auto stage_site = [this, &staged](const SiteId key) {
    if (staged.contains(key)) {
      return;
    }
    const auto accepted = timelines_.find(key);
    if (accepted == timelines_.end()) {
      staged.try_emplace(key);
    } else {
      staged.emplace(key, accepted->second);
    }
  };

  stage_site(position);
  for (const JumpEvent &event : path.events) {
    position = layout_.shifted(position, event.axis, static_cast<Coord>(event.direction));
    stage_site(position);
  }
}

void OccupancyIndex::adjust_path(TimelineMap &timelines, const ContinuousPath &path,
                                 const std::int64_t sign) const {
  if (sign != -1 && sign != 1) {
    throw std::invalid_argument("occupancy path adjustment sign must be -1 or +1");
  }
  path.validate(layout_.dimension());
  SiteId position = layout_.encode_covering(path.start);
  timeline(timelines, position).adjust_initial(sign);
  for (const JumpEvent &event : path.events) {
    const SiteId old_key = position;
    position = layout_.shifted(position, event.axis, static_cast<Coord>(event.direction));
    if (old_key == position) {
      continue;
    }
    timeline(timelines, old_key).adjust_event(event.time, -sign);
    timeline(timelines, position).adjust_event(event.time, sign);
  }
}

double OccupancyIndex::integrate_path_occupancy(TimelineMap &timelines,
                                                const ContinuousPath &path) const {
  path.validate(layout_.dimension());
  SiteId position = layout_.encode_covering(path.start);
  double previous = 0.0;
  double total = 0.0;
  for (const JumpEvent &event : path.events) {
    total += timeline(timelines, position).integral(previous, event.time);
    position = layout_.shifted(position, event.axis, static_cast<Coord>(event.direction));
    previous = event.time;
  }
  total += timeline(timelines, position).integral(previous, path.duration);
  return total;
}

void OccupancyIndex::rebuild(const std::span<const ContinuousPath> paths) {
  TimelineMap rebuilt;
  for (const ContinuousPath &path : paths) {
    adjust_path(rebuilt, path, 1);
  }
  timelines_.swap(rebuilt);
}

OccupancyIndex::ReplacementTransaction
OccupancyIndex::begin_replacement(const std::span<const PathReplacementView> replacements,
                                  const double current_overlap) {
  for (std::size_t index = 0; index < replacements.size(); ++index) {
    if (index != 0 && replacements[index - 1].label >= replacements[index].label) {
      throw std::invalid_argument("occupancy replacement labels must be sorted and unique");
    }
  }

  TimelineMap staged;
  for (const PathReplacementView &replacement : replacements) {
    stage_path_timelines(replacement.old_path, staged);
    stage_path_timelines(replacement.new_path, staged);
  }

  double removed = 0.0;
  for (const PathReplacementView &replacement : replacements) {
    removed +=
        integrate_path_occupancy(staged, replacement.old_path) - replacement.old_path.duration;
    adjust_path(staged, replacement.old_path, -1);
  }

  double added = 0.0;
  for (const PathReplacementView &replacement : replacements) {
    added += integrate_path_occupancy(staged, replacement.new_path);
    adjust_path(staged, replacement.new_path, 1);
  }
  return ReplacementTransaction(*this, current_overlap - removed + added, std::move(staged));
}

bool OccupancyIndex::same_occupancies(const TimelineMap &left, const TimelineMap &right) noexcept {
  auto left_entry = left.begin();
  auto right_entry = right.begin();
  while (true) {
    while (left_entry != left.end() && left_entry->second.empty()) {
      ++left_entry;
    }
    while (right_entry != right.end() && right_entry->second.empty()) {
      ++right_entry;
    }
    if (left_entry == left.end() || right_entry == right.end()) {
      return left_entry == left.end() && right_entry == right.end();
    }
    if (left_entry->first != right_entry->first ||
        left_entry->second.initial != right_entry->second.initial ||
        left_entry->second.deltas != right_entry->second.deltas) {
      return false;
    }
    ++left_entry;
    ++right_entry;
  }
}

bool OccupancyIndex::represents(const std::span<const ContinuousPath> paths) const {
  TimelineMap expected;
  for (const ContinuousPath &path : paths) {
    adjust_path(expected, path, 1);
  }
  return same_occupancies(timelines_, expected);
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
