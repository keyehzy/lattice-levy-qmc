#ifndef QMC_SRC_STITCH_PROPOSAL_HPP
#define QMC_SRC_STITCH_PROPOSAL_HPP

#include "qmc/continuous_configuration.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace qmc::detail {

class StitchSeamContext;

// Complete free conditional proposal for one fixed-seam stitch. The caller
// retains ownership of Metropolis evaluation, statistics, and transactional
// publication into the accepted chain state.
struct StitchProposal {
  std::vector<std::pair<ParticleId, ContinuousPath>> replacements;
  std::vector<ParticleId> successors;
  std::size_t successor_changes = 0;
};

[[nodiscard]] StitchProposal sample_stitch_proposal(const ContinuousConfiguration &configuration,
                                                    std::span<const ParticleId> strands,
                                                    StitchSeamContext &seam, Random &random);

} // namespace qmc::detail

#endif
