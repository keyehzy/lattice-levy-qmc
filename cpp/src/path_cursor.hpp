#ifndef QMC_SRC_PATH_CURSOR_HPP
#define QMC_SRC_PATH_CURSOR_HPP

#include "qmc/path.hpp"

#include <cstddef>
#include <cstdint>
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

enum class PathCutSide : std::uint8_t {
  BeforeEvents,
  ThroughEvents,
};

// A view of one interval in a source path. Its endpoint positions and event
// range reflect the PathCutSide values used to construct it.
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
  // The default right-continuous slice represents (left, right].
  [[nodiscard]] PathSlice slice(const PathCut &left, const PathCut &right) const;
  [[nodiscard]] PathSlice slice(const PathCut &left, const PathCut &right, PathCutSide left_side,
                                PathCutSide right_side) const;

private:
  const ContinuousPath &path_;
  std::size_t next_event_ = 0;
  Site position_;
  std::optional<PathCut> last_cut_;
};

// Copies a slice into a standalone path whose time origin is tau0.
[[nodiscard]] ContinuousPath materialize_path_slice(const PathSlice &slice);

// Joins the prefix before prefix_slice, replacement, and the suffix after
// suffix_slice. The replacement may end at a different covering representative
// from suffix_slice.end; the retained suffix is translated accordingly.
[[nodiscard]] ContinuousPath splice_path_slices(const PathSlice &prefix_slice,
                                                const PathSlice &suffix_slice,
                                                const ContinuousPath &replacement);

// Replaces the source interval represented by slice with a path on [0,
// tau1-tau0], retaining source events at tau0 and discarding source events at
// tau1.
[[nodiscard]] ContinuousPath replace_path_slice(const PathSlice &slice,
                                                const ContinuousPath &replacement);

// Places first at time zero and second at the right end of result_duration.
// second_translation must connect the covering endpoint of first to the start
// of second.
[[nodiscard]] ContinuousPath concatenate_path_slices(const PathSlice &first,
                                                     const PathSlice &second,
                                                     const Site &second_translation,
                                                     double result_duration);

} // namespace qmc::detail

#endif
