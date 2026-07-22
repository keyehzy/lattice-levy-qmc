#ifndef QMC_CONTINUOUS_OBSERVABLES_HPP
#define QMC_CONTINUOUS_OBSERVABLES_HPP

#include "qmc/continuous_configuration.hpp"
#include "qmc/torus_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qmc {

// One physical nearest-neighbor hop in a continuous configuration. Time is in
// [0, beta], sites are reduced onto the torus, and direction retains the
// covering-space sign even when periodic reduction makes both signs coincide.
struct ContinuousHop {
  double time;
  ParticleId particle;
  SiteId departure;
  SiteId arrival;
  Axis axis;
  std::int8_t direction;

  [[nodiscard]] bool operator==(const ContinuousHop &) const = default;
};

// Owning event geometry derived from one valid continuous configuration.
// seam_positions() is the state immediately before events at time zero. Hops
// are globally nondecreasing in time; ties retain particle/path event order and
// are exposed as atomic groups.
class ContinuousMeasurementContext {
public:
  explicit ContinuousMeasurementContext(const ContinuousConfiguration &configuration);

  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] const TorusLayout &layout() const noexcept { return layout_; }
  [[nodiscard]] std::span<const SiteId> seam_positions() const noexcept { return seam_positions_; }
  [[nodiscard]] std::span<const ContinuousHop> hops() const noexcept { return hops_; }
  [[nodiscard]] std::size_t event_group_count() const noexcept {
    return event_group_offsets_.size() - 1;
  }
  // Both group accessors throw std::out_of_range for an unknown group.
  [[nodiscard]] double event_time(std::size_t group) const;
  [[nodiscard]] std::span<const ContinuousHop> hops_at(std::size_t group) const;

private:
  Model model_;
  TorusLayout layout_;
  std::vector<SiteId> seam_positions_;
  std::vector<ContinuousHop> hops_;
  // Includes the terminal offset, so an empty context owns the single value 0.
  std::vector<std::size_t> event_group_offsets_{0};
};

} // namespace qmc

#endif
