#ifndef QMC_SRC_STITCH_SEAM_CONTEXT_HPP
#define QMC_SRC_STITCH_SEAM_CONTEXT_HPP

#include "qmc/continuous_configuration.hpp"
#include "qmc/free_numerics.hpp"
#include "qmc/torus_layout.hpp"
#include "torus_bridge_distribution.hpp"

#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

namespace qmc::detail {

using StitchPartnerBuckets = std::unordered_map<SiteId, std::vector<ParticleId>, SiteIdHash>;

// State shared by stitch attempts at one fixed time window. An accepted stitch
// preserves every path's position at tau0 and permutes the physical positions
// at tau1, so the left-position buckets and displacement-keyed bridge cache
// remain valid for the complete fixed-seam sweep.
class StitchSeamContext {
public:
  StitchSeamContext(const ContinuousConfiguration &configuration, double tau0, double tau1,
                    const TorusLayout &layout, const FreePathKernels &kernels);

  [[nodiscard]] double tau0() const noexcept { return tau0_; }
  [[nodiscard]] double tau1() const noexcept { return tau1_; }
  [[nodiscard]] double duration() const noexcept { return tau1_ - tau0_; }
  [[nodiscard]] std::span<const SiteId> left_site_ids() const noexcept { return left_site_ids_; }
  [[nodiscard]] const StitchPartnerBuckets &partner_buckets() const noexcept {
    return partner_buckets_;
  }

  [[nodiscard]] const TorusBridgeDistribution &bridge_distribution(const Site &left,
                                                                   const Site &right);

  // Diagnostic hooks for the fixed-seam invariant and focused cache tests.
  [[nodiscard]] bool matches_left_positions(const ContinuousConfiguration &configuration) const;
  [[nodiscard]] std::size_t cached_distribution_count() const noexcept {
    return bridge_distributions_.size();
  }

private:
  [[nodiscard]] std::vector<SiteId>
  encode_left_positions(const ContinuousConfiguration &configuration) const;

  double tau0_;
  double tau1_;
  const TorusLayout &layout_;
  const FreePathKernels &kernels_;
  std::vector<SiteId> left_site_ids_;
  StitchPartnerBuckets partner_buckets_;
  std::unordered_map<SiteId, TorusBridgeDistribution, SiteIdHash> bridge_distributions_;
};

} // namespace qmc::detail

#endif
