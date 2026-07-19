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

} // namespace

PathCursor::PathCursor(const ContinuousPath &path) : path_(path), position_(path.start) {}

PathCut PathCursor::cut(const double tau) {
  if (!std::isfinite(tau) || tau < 0.0 || tau > path_.duration) {
    throw std::invalid_argument("path cut must lie in [0, duration]");
  }
  if (last_cut_.has_value() && tau < last_cut_->time) {
    throw std::logic_error("path cursor cuts must be nondecreasing");
  }
  if (last_cut_.has_value() && tau == last_cut_->time) {
    return *last_cut_;
  }

  while (next_event_ < path_.events.size() && path_.events[next_event_].time < tau) {
    apply_event(position_, path_.events[next_event_]);
    ++next_event_;
  }
  const std::size_t events_before = next_event_;
  const Site position_before = position_;
  while (next_event_ < path_.events.size() && path_.events[next_event_].time == tau) {
    apply_event(position_, path_.events[next_event_]);
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
  if (left.source != &path_ || right.source != &path_ || left.time > right.time ||
      left.events_through > right.events_through || right.events_through > path_.events.size()) {
    throw std::logic_error("invalid path slice boundaries");
  }
  return PathSlice{
      .source = path_,
      .tau0 = left.time,
      .tau1 = right.time,
      .start = left.position_through,
      .end = right.position_through,
      .first_event = left.events_through,
      .events = std::span<const JumpEvent>(path_.events)
                    .subspan(left.events_through, right.events_through - left.events_through),
  };
}

ContinuousPath materialize_path_slice(const PathSlice &slice) {
  std::vector<JumpEvent> events;
  events.reserve(slice.events.size());
  for (const JumpEvent &event : slice.events) {
    events.push_back(JumpEvent{
        .time = event.time - slice.tau0,
        .axis = event.axis,
        .direction = event.direction,
    });
  }

  ContinuousPath result{
      .duration = slice.tau1 - slice.tau0,
      .start = slice.start,
      .end = slice.end,
      .events = std::move(events),
  };
  result.validate(result.start.size());
  return result;
}

ContinuousPath replace_path_slice(const PathSlice &slice, const ContinuousPath &replacement) {
  if (replacement.duration != slice.tau1 - slice.tau0 || replacement.start != slice.start ||
      replacement.end != slice.end) {
    throw std::invalid_argument("replacement path does not match slice endpoints and duration");
  }

  const ContinuousPath &source = slice.source;
  const std::size_t suffix_event =
      checked_add_size(slice.first_event, slice.events.size(), "path slice event range overflowed");
  if (suffix_event > source.events.size()) {
    throw std::logic_error("path slice event range exceeds its source");
  }
  const std::size_t retained_events = source.events.size() - slice.events.size();
  const std::size_t result_size = checked_add_size(retained_events, replacement.events.size(),
                                                   "replacement path event count exceeds size_t");

  std::vector<JumpEvent> events;
  events.reserve(result_size);
  const std::span<const JumpEvent> source_events(source.events);
  const std::span<const JumpEvent> prefix = source_events.first(slice.first_event);
  events.insert(events.end(), prefix.begin(), prefix.end());
  for (const JumpEvent &event : replacement.events) {
    events.push_back(JumpEvent{
        .time = slice.tau0 + event.time,
        .axis = event.axis,
        .direction = event.direction,
    });
  }
  const std::span<const JumpEvent> suffix = source_events.subspan(suffix_event);
  events.insert(events.end(), suffix.begin(), suffix.end());

  ContinuousPath result{
      .duration = source.duration,
      .start = source.start,
      .end = source.end,
      .events = std::move(events),
  };
  result.validate(result.start.size());
  return result;
}

} // namespace qmc::detail
