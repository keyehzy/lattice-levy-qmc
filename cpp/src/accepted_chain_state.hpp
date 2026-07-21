#ifndef QMC_SRC_ACCEPTED_CHAIN_STATE_HPP
#define QMC_SRC_ACCEPTED_CHAIN_STATE_HPP

#include "occupancy_index.hpp"
#include "qmc/continuous_configuration.hpp"
#include "qmc/permutation.hpp"
#include "qmc/torus_layout.hpp"

#include <optional>
#include <vector>

namespace qmc::detail {

// One accepted Markov-chain state and the derived occupancy data that belongs
// to it. The independent full overlap evaluator remains the audit oracle; live
// proposals update this owner transactionally through the sparse ledger.
class AcceptedChainState {
public:
  struct PathReplacement {
    ParticleId label;
    ContinuousPath path;
  };

  class ReplacementTransaction {
  public:
    ReplacementTransaction(const ReplacementTransaction &) = delete;
    ReplacementTransaction &operator=(const ReplacementTransaction &) = delete;
    ReplacementTransaction(ReplacementTransaction &&other) noexcept;
    ReplacementTransaction &operator=(ReplacementTransaction &&) = delete;
    ~ReplacementTransaction() = default;

    [[nodiscard]] double proposed_overlap() const noexcept { return proposed_overlap_; }

    // Publishes paths, topology, ledger, and cached overlap as one non-throwing
    // accepted-state update. Abandonment leaves the accepted owner untouched.
    void commit() noexcept;

  private:
    friend class AcceptedChainState;

    ReplacementTransaction(AcceptedChainState &owner, std::vector<PathReplacement> replacements,
                           std::optional<Permutation> topology, double proposed_overlap,
                           OccupancyIndex::ReplacementTransaction occupancy_transaction) noexcept;

    AcceptedChainState *owner_;
    std::vector<PathReplacement> replacements_;
    std::optional<Permutation> topology_;
    double proposed_overlap_;
    OccupancyIndex::ReplacementTransaction occupancy_transaction_;
  };

  explicit AcceptedChainState(ContinuousConfiguration configuration);

  AcceptedChainState(const AcceptedChainState &) = default;
  AcceptedChainState &operator=(const AcceptedChainState &) = default;
  AcceptedChainState(AcceptedChainState &&) noexcept = default;
  AcceptedChainState &operator=(AcceptedChainState &&) noexcept = default;

  [[nodiscard]] const ContinuousConfiguration &configuration() const noexcept {
    return configuration_;
  }
  [[nodiscard]] double pair_overlap() const noexcept { return pair_overlap_; }

  [[nodiscard]] ReplacementTransaction
  begin_replacement(std::vector<PathReplacement> replacements,
                    std::optional<std::vector<ParticleId>> successors);

  // Expensive audit seams used by invariant tests.
  [[nodiscard]] bool occupancy_matches_configuration() const;
  [[nodiscard]] double occupancy_pair_overlap();

private:
  [[nodiscard]] const ContinuousPath &
  path_after_replacement(ParticleId label, const std::vector<PathReplacement> &replacements) const;
  void validate_replacement_inputs(const std::vector<PathReplacement> &replacements) const;
  [[nodiscard]] bool
  replacement_endpoints_unchanged(const std::vector<PathReplacement> &replacements) const;
  void validate_replacement_connectivity(const std::vector<PathReplacement> &replacements,
                                         const Permutation &topology, bool topology_changed) const;
  void validate_replacement_state(const std::vector<PathReplacement> &replacements,
                                  const std::optional<Permutation> &topology) const;
  void publish(ReplacementTransaction &transaction) noexcept;

  ContinuousConfiguration configuration_;
  TorusLayout layout_;
  OccupancyIndex occupancy_;
  double pair_overlap_ = 0.0;
};

} // namespace qmc::detail

#endif
