#ifndef QMC_SRC_CONTINUOUS_DETAIL_HPP
#define QMC_SRC_CONTINUOUS_DETAIL_HPP

#include "qmc/free_boson.hpp"
#include "qmc/model.hpp"
#include "qmc/path.hpp"
#include "qmc/random.hpp"

#include <vector>

namespace qmc::detail {

// Returned paths have the same order as labels.
[[nodiscard]] std::vector<ContinuousPath> sample_paths_for_cycle(const Cycle &labels,
                                                                 const Model &model,
                                                                 const FreePathKernels &kernels,
                                                                 Random &random);

} // namespace qmc::detail

#endif
