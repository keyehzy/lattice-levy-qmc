#ifndef QMC_SRC_CONTINUOUS_EVENT_SWEEP_HPP
#define QMC_SRC_CONTINUOUS_EVENT_SWEEP_HPP

#include "qmc/continuous_observables.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qmc::detail {

struct ContinuousEventSweepData {
  std::vector<SiteId> seam_positions;
  std::vector<ContinuousHop> hops;
  std::vector<std::size_t> group_offsets;
};

// Collects path-local hop geometry, then establishes the one authoritative
// global time order and equal-time grouping used by continuous measurements
// and full-state interaction evaluators.
[[nodiscard]] ContinuousEventSweepData
build_continuous_event_sweep(const ContinuousConfiguration &configuration,
                             const TorusLayout &layout);

// Equivalent construction over an arbitrary label-indexed path view. This is
// the shared boundary used by full-state audit evaluators that do not own a
// ContinuousConfiguration.
[[nodiscard]] ContinuousEventSweepData
build_continuous_event_sweep(const Model &model, std::span<const ContinuousPath *const> worldlines,
                             const TorusLayout &layout);

} // namespace qmc::detail

#endif
