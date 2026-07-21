#ifndef QMC_SRC_STITCH_SELECTION_HPP
#define QMC_SRC_STITCH_SELECTION_HPP

#include "qmc/model.hpp"
#include "qmc/random.hpp"
#include "qmc/torus_layout.hpp"
#include "stitch_matching.hpp"
#include "stitch_seam_context.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace qmc::detail {

class SelectedStitchStrands;

[[nodiscard]] SelectedStitchStrands
select_stitch_strands(ParticleId anchor, std::size_t strand_count,
                      std::span<const SiteId> positions, const StitchPartnerBuckets &buckets,
                      const TorusLayout &layout, std::size_t locality_radius,
                      double global_partner_probability, Random &random);

// Fixed-capacity result for one automatic stitch-strand selection. The public
// stitch limit makes heap-backed output and an N-sized selected-label mask
// unnecessary.
class SelectedStitchStrands {
public:
  [[nodiscard]] std::span<const ParticleId> strands() const noexcept {
    return std::span(labels_).first(size_);
  }

private:
  SelectedStitchStrands() = default;

  friend SelectedStitchStrands
  select_stitch_strands(ParticleId anchor, std::size_t strand_count,
                        std::span<const SiteId> positions, const StitchPartnerBuckets &buckets,
                        const TorusLayout &layout, std::size_t locality_radius,
                        double global_partner_probability, Random &random);

  std::array<ParticleId, kMaxStitchStrands> labels_{};
  std::size_t size_ = 0;
};

} // namespace qmc::detail

#endif
