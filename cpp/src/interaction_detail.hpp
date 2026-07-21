#ifndef QMC_SRC_INTERACTION_DETAIL_HPP
#define QMC_SRC_INTERACTION_DETAIL_HPP

#include "qmc/interaction.hpp"
#include "qmc/model.hpp"
#include "qmc/path.hpp"

#include <span>

namespace qmc::detail {

// Reference action evaluator over an arbitrary label-indexed path view. This
// permits transactional proposal evaluation without mutating the accepted state.
[[nodiscard]] double pair_overlap_time_for_paths(const Model &model,
                                                 std::span<const ContinuousPath *const> worldlines);

// Validates the public configuration/model provenance boundary before an
// interaction measurement performs numerical work.
void validate_interaction_model_provenance(const ContinuousConfiguration &configuration,
                                           const InteractingModel &model);

// Shared scalar calculation for a previously validated configuration/model
// pair. The sampler uses this with its accepted-state overlap cache.
[[nodiscard]] InteractionMeasurement
interaction_measurement_from_validated_overlap(const ContinuousConfiguration &configuration,
                                               const InteractingModel &model, double pair_overlap);

} // namespace qmc::detail

#endif
