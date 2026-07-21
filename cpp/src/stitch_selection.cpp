#include "stitch_selection.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace qmc::detail {
namespace {

bool neighborhood_is_small(const TorusLayout &layout, const std::size_t locality_radius,
                           const std::size_t occupied_sites) {
  const auto linear_size = static_cast<std::size_t>(layout.linear_size());
  if (locality_radius >= (linear_size / 2) + (linear_size % 2)) {
    return false;
  }
  const std::size_t width = (2 * locality_radius) + 1;
  const std::size_t limit = occupied_sites > (std::numeric_limits<std::size_t>::max() - 1) / 4
                                ? std::numeric_limits<std::size_t>::max()
                                : (4 * occupied_sites) + 1;
  std::size_t volume = 1;
  for (std::size_t axis = 0; axis < layout.dimension(); ++axis) {
    if (volume > limit / width) {
      return false;
    }
    volume *= width;
  }
  return true;
}

std::vector<ParticleId> same_site_candidates(const ParticleId particle, const SiteId position,
                                             const StitchPartnerBuckets &buckets) {
  std::vector<ParticleId> candidates;
  const auto bucket = buckets.find(position);
  if (bucket == buckets.end()) {
    return candidates;
  }
  for (const ParticleId label : bucket->second) {
    if (label != particle) {
      candidates.push_back(label);
    }
  }
  return candidates;
}

std::vector<ParticleId> scan_neighbor_candidates(const ParticleId particle, const SiteId position,
                                                 const StitchPartnerBuckets &buckets,
                                                 const TorusLayout &layout,
                                                 const std::size_t locality_radius) {
  std::vector<ParticleId> candidates;
  for (const auto &[site, labels] : buckets) {
    if (!layout.within_radius(position, site, locality_radius)) {
      continue;
    }
    for (const ParticleId label : labels) {
      if (label != particle) {
        candidates.push_back(label);
      }
    }
  }
  return candidates;
}

std::vector<ParticleId> local_stitch_candidates(const ParticleId particle,
                                                const std::span<const SiteId> positions,
                                                const StitchPartnerBuckets &buckets,
                                                const TorusLayout &layout,
                                                const std::size_t locality_radius) {
  if (locality_radius == 0) {
    return same_site_candidates(particle, positions[particle], buckets);
  }
  if (!neighborhood_is_small(layout, locality_radius, buckets.size())) {
    return scan_neighbor_candidates(particle, positions[particle], buckets, layout,
                                    locality_radius);
  }

  std::vector<ParticleId> candidates;
  for (const SiteId neighbor :
       layout.neighbors_within_radius(positions[particle], locality_radius)) {
    const auto bucket = buckets.find(neighbor);
    if (bucket == buckets.end()) {
      continue;
    }
    for (const ParticleId label : bucket->second) {
      if (label != particle) {
        candidates.push_back(label);
      }
    }
  }
  return candidates;
}

bool is_selected(const std::span<const ParticleId> selected, const ParticleId candidate) {
  return std::ranges::find(selected, candidate) != selected.end();
}

std::optional<ParticleId> draw_uniform_unselected(const std::span<const ParticleId> pool,
                                                  const std::span<const ParticleId> selected,
                                                  Random &random) {
  std::size_t candidate_count = 0;
  for (const ParticleId label : pool) {
    if (!is_selected(selected, label)) {
      ++candidate_count;
    }
  }
  if (candidate_count == 0) {
    return std::nullopt;
  }

  auto draw =
      static_cast<std::size_t>(random.uniform_index(static_cast<std::uint64_t>(candidate_count)));
  for (const ParticleId label : pool) {
    if (is_selected(selected, label)) {
      continue;
    }
    if (draw == 0) {
      return label;
    }
    --draw;
  }
  throw std::logic_error("failed to draw a local stitch strand");
}

ParticleId draw_uniform_unselected(const std::size_t particle_count,
                                   const std::span<const ParticleId> selected, Random &random) {
  auto draw = static_cast<std::size_t>(
      random.uniform_index(static_cast<std::uint64_t>(particle_count - selected.size())));
  for (std::size_t label = 0; label < particle_count; ++label) {
    const auto candidate = static_cast<ParticleId>(label);
    if (is_selected(selected, candidate)) {
      continue;
    }
    if (draw == 0) {
      return candidate;
    }
    --draw;
  }
  throw std::logic_error("failed to draw an unselected stitch strand");
}

} // namespace

SelectedStitchStrands
select_stitch_strands(const ParticleId anchor, const std::size_t strand_count,
                      const std::span<const SiteId> positions, const StitchPartnerBuckets &buckets,
                      const TorusLayout &layout, const std::size_t locality_radius,
                      const double global_partner_probability, Random &random) {
  assert(std::isfinite(global_partner_probability) && global_partner_probability >= 0.0 &&
         global_partner_probability <= 1.0);
  assert(strand_count >= 2 && strand_count <= kMaxStitchStrands &&
         strand_count <= positions.size());
  assert(static_cast<std::size_t>(anchor) < positions.size());

  const std::vector<ParticleId> local_pool =
      local_stitch_candidates(anchor, positions, buckets, layout, locality_radius);
  SelectedStitchStrands selected;
  selected.labels_[0] = anchor;
  selected.size_ = 1;

  while (selected.size_ < strand_count) {
    const bool use_global = random.uniform_unit() < global_partner_probability;
    const std::optional<ParticleId> local =
        use_global ? std::nullopt : draw_uniform_unselected(local_pool, selected.strands(), random);
    const ParticleId next =
        local.has_value() ? *local
                          : draw_uniform_unselected(positions.size(), selected.strands(), random);
    selected.labels_[selected.size_] = next;
    ++selected.size_;
  }
  return selected;
}

} // namespace qmc::detail
