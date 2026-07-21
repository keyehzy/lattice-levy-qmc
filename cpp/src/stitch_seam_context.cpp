#include "stitch_seam_context.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qmc::detail {

StitchSeamContext::StitchSeamContext(const ContinuousConfiguration &configuration,
                                     const double tau0, const double tau1,
                                     const TorusLayout &layout, const FreePathKernels &kernels)
    : tau0_(tau0), tau1_(tau1), layout_(layout), kernels_(kernels) {
  if (!std::isfinite(tau0_) || !std::isfinite(tau1_) || tau0_ < 0.0 ||
      tau1_ > configuration.model().beta() || tau1_ <= tau0_) {
    throw std::invalid_argument("stitch seam must satisfy 0 <= tau0 < tau1 <= beta");
  }
  if (configuration.model().linear_size() != layout_.linear_size() ||
      configuration.model().dimension() != layout_.dimension()) {
    throw std::invalid_argument("stitch seam layout does not match the configuration");
  }
  const TorusLayout *const kernel_layout = kernels_.torus_layout();
  if (kernel_layout == nullptr || *kernel_layout != layout_) {
    throw std::invalid_argument("stitch seam kernels do not match the torus layout");
  }

  left_site_ids_ = encode_left_positions(configuration);
  partner_buckets_.reserve(left_site_ids_.size());
  for (std::size_t label = 0; label < left_site_ids_.size(); ++label) {
    partner_buckets_[left_site_ids_[label]].push_back(static_cast<ParticleId>(label));
  }
  bridge_distributions_.reserve(std::min<std::size_t>(layout_.volume(), 64));
}

std::vector<SiteId>
StitchSeamContext::encode_left_positions(const ContinuousConfiguration &configuration) const {
  const std::vector<Site> positions = configuration.positions_at(tau0_);
  std::vector<SiteId> site_ids;
  site_ids.reserve(positions.size());
  for (const Site &position : positions) {
    site_ids.push_back(layout_.encode(position));
  }
  return site_ids;
}

const TorusBridgeDistribution &StitchSeamContext::bridge_distribution(const Site &left,
                                                                      const Site &right) {
  const SiteId displacement =
      layout_.flat_displacement(layout_.encode_covering(left), layout_.encode_covering(right));
  const auto [distribution, inserted] =
      bridge_distributions_.try_emplace(displacement, displacement, duration(), kernels_);
  static_cast<void>(inserted);
  return distribution->second;
}

bool StitchSeamContext::matches_left_positions(const ContinuousConfiguration &configuration) const {
  if (configuration.model().linear_size() != layout_.linear_size() ||
      configuration.model().dimension() != layout_.dimension() ||
      configuration.model().beta() < tau0_) {
    return false;
  }
  return encode_left_positions(configuration) == left_site_ids_;
}

} // namespace qmc::detail
