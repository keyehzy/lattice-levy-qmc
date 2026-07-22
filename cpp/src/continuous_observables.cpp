#include "qmc/continuous_observables.hpp"

#include "continuous_event_sweep.hpp"

#include <stdexcept>
#include <utility>

namespace qmc {

ContinuousMeasurementContext::ContinuousMeasurementContext(
    const ContinuousConfiguration &configuration)
    : model_(configuration.model()), layout_(model_.linear_size(), model_.dimension()) {
  detail::ContinuousEventSweepData data =
      detail::build_continuous_event_sweep(configuration, layout_);
  seam_positions_ = std::move(data.seam_positions);
  hops_ = std::move(data.hops);
  event_group_offsets_ = std::move(data.group_offsets);
}

double ContinuousMeasurementContext::event_time(const std::size_t group) const {
  if (group >= event_group_count()) {
    throw std::out_of_range("continuous event group index is out of range");
  }
  return hops_[event_group_offsets_[group]].time;
}

std::span<const ContinuousHop>
ContinuousMeasurementContext::hops_at(const std::size_t group) const {
  if (group >= event_group_count()) {
    throw std::out_of_range("continuous event group index is out of range");
  }
  const std::size_t begin = event_group_offsets_[group];
  const std::size_t end = event_group_offsets_[group + 1];
  return std::span<const ContinuousHop>(hops_).subspan(begin, end - begin);
}

} // namespace qmc
