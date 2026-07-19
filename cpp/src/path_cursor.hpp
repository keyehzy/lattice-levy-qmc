#ifndef QMC_SRC_PATH_CURSOR_HPP
#define QMC_SRC_PATH_CURSOR_HPP

#include "qmc/path.hpp"

#include <cstddef>
#include <optional>
#include <span>

namespace qmc::detail {

// State at one time cut. Positions before/through the cut are the left and
// right limits respectively; events_through includes every event at the cut.
struct PathCut {
  const ContinuousPath *source;
  double time;
  Site position_before;
  Site position_through;
  std::size_t events_before;
  std::size_t events_through;
};

// A view of the right-continuous interval (tau0, tau1] in a source path.
// The endpoint positions include all events at their respective cuts.
struct PathSlice {
  const ContinuousPath &source;
  double tau0;
  double tau1;
  Site start;
  Site end;
  std::size_t first_event;
  std::span<const JumpEvent> events;
};

// Traverses an already validated, unchanged path at nondecreasing cuts. Each
// event is visited at most once across the cursor's lifetime.
class PathCursor {
public:
  explicit PathCursor(const ContinuousPath &path);

  [[nodiscard]] PathCut cut(double tau);
  [[nodiscard]] PathSlice slice(const PathCut &left, const PathCut &right) const;

private:
  const ContinuousPath &path_;
  std::size_t next_event_ = 0;
  Site position_;
  std::optional<PathCut> last_cut_;
};

// Copies a slice into a standalone path whose time origin is tau0.
[[nodiscard]] ContinuousPath materialize_path_slice(const PathSlice &slice);

// Replaces the source interval represented by slice with a path on [0,
// tau1-tau0], retaining source events at tau0 and discarding source events at
// tau1.
[[nodiscard]] ContinuousPath replace_path_slice(const PathSlice &slice,
                                                const ContinuousPath &replacement);

} // namespace qmc::detail

#endif
