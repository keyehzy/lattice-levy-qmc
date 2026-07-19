#ifndef QMC_SRC_INTERACTION_DETAIL_HPP
#define QMC_SRC_INTERACTION_DETAIL_HPP

#include "qmc/model.hpp"
#include "qmc/path.hpp"

#include <span>

namespace qmc::detail {

// Reference action evaluator over an arbitrary label-indexed path view. This
// permits transactional proposal evaluation without mutating the accepted state.
[[nodiscard]] double pair_overlap_time_for_paths(const Model &model,
                                                 std::span<const ContinuousPath *const> worldlines);

} // namespace qmc::detail

#endif
