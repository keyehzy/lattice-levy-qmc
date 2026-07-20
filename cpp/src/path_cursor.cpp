#include "path_cursor.hpp"

#include "checked_math.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc::detail {
namespace {

void apply_event(Site &position, const JumpEvent &event) {
  position[event.axis] = checked_add(position[event.axis], static_cast<Coord>(event.direction),
                                     "continuous path coordinate exceeds int64 range");
}

std::size_t event_boundary(const PathCut &cut, const PathCutSide side) {
  return side == PathCutSide::BeforeEvents ? cut.events_before : cut.events_through;
}

const Site &cut_position(const PathCut &cut, const PathCutSide side) {
  return side == PathCutSide::BeforeEvents ? cut.position_before : cut.position_through;
}

void append_shifted_events(std::vector<JumpEvent> &destination,
                           const std::span<const JumpEvent> source, const double shift) {
  for (const JumpEvent &event : source) {
    const double time = event.time + shift;
    if (!std::isfinite(time)) {
      throw std::overflow_error("shifted path event time overflowed");
    }
    destination.push_back(JumpEvent{
        .time = time,
        .axis = event.axis,
        .direction = event.direction,
    });
  }
}

std::size_t slice_end_event(const PathSlice &slice) {
  return checked_add_size(slice.first_event, slice.events.size(),
                          "path slice event range overflowed");
}

} // namespace

PathCursor::PathCursor(const ContinuousPath &path) : path_(path), position_(path.start()) {}

PathCut PathCursor::cut(const double tau) {
  if (!std::isfinite(tau) || tau < 0.0 || tau > path_.duration()) {
    throw std::invalid_argument("path cut must lie in [0, duration]");
  }
  if (last_cut_.has_value() && tau < last_cut_->time) {
    throw std::logic_error("path cursor cuts must be nondecreasing");
  }
  if (last_cut_.has_value() && tau == last_cut_->time) {
    return *last_cut_;
  }

  while (next_event_ < path_.events().size() && path_.events()[next_event_].time < tau) {
    apply_event(position_, path_.events()[next_event_]);
    ++next_event_;
  }
  const std::size_t events_before = next_event_;
  const Site position_before = position_;
  while (next_event_ < path_.events().size() && path_.events()[next_event_].time == tau) {
    apply_event(position_, path_.events()[next_event_]);
    ++next_event_;
  }

  PathCut result{
      .source = &path_,
      .time = tau,
      .position_before = position_before,
      .position_through = position_,
      .events_before = events_before,
      .events_through = next_event_,
  };
  last_cut_ = result;
  return result;
}

PathSlice PathCursor::slice(const PathCut &left, const PathCut &right) const {
  return slice(left, right, PathCutSide::ThroughEvents, PathCutSide::ThroughEvents);
}

PathSlice PathCursor::slice(const PathCut &left, const PathCut &right, const PathCutSide left_side,
                            const PathCutSide right_side) const {
  if (left.source != &path_ || right.source != &path_ || left.time > right.time ||
      left.events_through > right.events_through || right.events_through > path_.events().size()) {
    throw std::logic_error("invalid path slice boundaries");
  }
  const std::size_t first_event = event_boundary(left, left_side);
  const std::size_t last_event = event_boundary(right, right_side);
  if (first_event > last_event || last_event > path_.events().size()) {
    throw std::logic_error("path slice endpoint sides select an invalid event range");
  }
  return PathSlice{
      .source = path_,
      .tau0 = left.time,
      .tau1 = right.time,
      .start = cut_position(left, left_side),
      .end = cut_position(right, right_side),
      .first_event = first_event,
      .events = path_.events().subspan(first_event, last_event - first_event),
  };
}

ContinuousPath materialize_path_slice(const PathSlice &slice) {
  std::vector<JumpEvent> events;
  events.reserve(slice.events.size());
  append_shifted_events(events, slice.events, -slice.tau0);

  return {slice.tau1 - slice.tau0, slice.start, slice.end, std::move(events)};
}

ContinuousPath splice_path_slices(const PathSlice &prefix_slice, const PathSlice &suffix_slice,
                                  const ContinuousPath &replacement) {
  if (prefix_slice.source.duration() != suffix_slice.source.duration() ||
      prefix_slice.tau0 != suffix_slice.tau0 || prefix_slice.tau1 != suffix_slice.tau1 ||
      replacement.duration() != prefix_slice.tau1 - prefix_slice.tau0 ||
      replacement.start() != prefix_slice.start ||
      replacement.end().size() != suffix_slice.end.size()) {
    throw std::invalid_argument("spliced path slices, replacement, and window do not match");
  }

  const ContinuousPath &prefix_source = prefix_slice.source;
  const ContinuousPath &suffix_source = suffix_slice.source;
  const std::size_t suffix_event = slice_end_event(suffix_slice);
  if (prefix_slice.first_event > prefix_source.events().size() ||
      suffix_event > suffix_source.events().size()) {
    throw std::logic_error("spliced path slice event range exceeds its source");
  }
  const std::size_t suffix_count = suffix_source.events().size() - suffix_event;
  std::size_t result_size = checked_add_size(prefix_slice.first_event, replacement.events().size(),
                                             "spliced path event count exceeds size_t");
  result_size =
      checked_add_size(result_size, suffix_count, "spliced path event count exceeds size_t");

  std::vector<JumpEvent> events;
  events.reserve(result_size);
  const std::span<const JumpEvent> prefix = prefix_source.events().first(prefix_slice.first_event);
  events.insert(events.end(), prefix.begin(), prefix.end());
  append_shifted_events(events, replacement.events(), prefix_slice.tau0);
  const std::span<const JumpEvent> suffix = suffix_source.events().subspan(suffix_event);
  events.insert(events.end(), suffix.begin(), suffix.end());

  Site end(replacement.end().size());
  for (std::size_t axis = 0; axis < end.size(); ++axis) {
    const Coord suffix_displacement =
        checked_subtract(suffix_source.end()[axis], suffix_slice.end[axis],
                         "spliced suffix displacement exceeds int64 range");
    end[axis] = checked_add(replacement.end()[axis], suffix_displacement,
                            "spliced path endpoint exceeds int64 range");
  }
  return {prefix_source.duration(), prefix_source.start(), std::move(end), std::move(events)};
}

ContinuousPath replace_path_slice(const PathSlice &slice, const ContinuousPath &replacement) {
  if (replacement.end() != slice.end) {
    throw std::invalid_argument("replacement path does not match slice endpoint");
  }
  return splice_path_slices(slice, slice, replacement);
}

ContinuousPath concatenate_path_slices(const PathSlice &first, const PathSlice &second,
                                       const Site &second_translation,
                                       const double result_duration) {
  if (!std::isfinite(result_duration) || result_duration < 0.0 ||
      first.start.size() != first.end.size() || first.end.size() != second.start.size() ||
      second.start.size() != second.end.size() || second.end.size() != second_translation.size()) {
    throw std::invalid_argument("concatenated path slices have incompatible durations or shapes");
  }

  Site translated_start(second.start.size());
  Site end(second.end.size());
  for (std::size_t axis = 0; axis < second.start.size(); ++axis) {
    translated_start[axis] = checked_add(second.start[axis], second_translation[axis],
                                         "concatenated path start exceeds int64 range");
    end[axis] = checked_add(second.end[axis], second_translation[axis],
                            "concatenated path endpoint exceeds int64 range");
  }
  if (first.end != translated_start) {
    throw std::invalid_argument("concatenated path slices do not meet in covering space");
  }

  const double second_duration = second.tau1 - second.tau0;
  const double second_offset = result_duration - second_duration;
  if (!std::isfinite(second_duration) || second_duration < 0.0 || !std::isfinite(second_offset) ||
      second_offset < 0.0) {
    throw std::invalid_argument("concatenated path slice durations exceed the result duration");
  }
  const std::size_t event_count = checked_add_size(first.events.size(), second.events.size(),
                                                   "concatenated path event count exceeds size_t");
  std::vector<JumpEvent> events;
  events.reserve(event_count);
  append_shifted_events(events, first.events, -first.tau0);
  append_shifted_events(events, second.events, second_offset - second.tau0);

  return {result_duration, first.start, std::move(end), std::move(events)};
}

} // namespace qmc::detail
