#include "continuous_event_sweep.hpp"

#include "checked_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmc::detail {
namespace {

double normalize_event_time(const double time, const double beta) {
  if (time >= 0.0 && time <= beta) {
    return time;
  }
  const double scale = std::max(std::abs(time), std::abs(beta));
  const double tolerance = 16.0 * std::numeric_limits<double>::epsilon() * scale;
  if (time > beta && time - beta <= tolerance) {
    return beta;
  }
  throw std::logic_error("continuous event time lies outside [0, beta]");
}

void require_vector_capacity(const std::size_t count, const std::size_t maximum,
                             const char *description) {
  if (count > maximum) {
    throw std::length_error(description);
  }
}

template <class PathAt>
ContinuousEventSweepData
build_continuous_event_sweep_impl(const Model &model, const std::size_t path_count, PathAt path_at,
                                  const TorusLayout &layout) {
  if (model.beta() <= 0.0) {
    throw std::invalid_argument("continuous event sweep requires positive beta");
  }
  if (path_count != model.particle_count()) {
    throw std::logic_error("continuous event-sweep path count does not match particle_count");
  }

  ContinuousEventSweepData data;
  require_vector_capacity(path_count, data.seam_positions.max_size(),
                          "continuous seam-position count exceeds vector capacity");

  std::size_t event_count = 0;
  for (std::size_t particle_index = 0; particle_index < path_count; ++particle_index) {
    const ContinuousPath *path = path_at(particle_index);
    if (path == nullptr) {
      throw std::logic_error("continuous event-sweep path view contains null");
    }
    if (path->dimension() != model.dimension()) {
      throw std::invalid_argument("continuous event-sweep path dimension does not match the model");
    }
    event_count = checked_add_size(event_count, path->events().size(),
                                   "continuous event count exceeds size_t");
  }
  require_vector_capacity(event_count, data.hops.max_size(),
                          "continuous hop count exceeds vector capacity");
  data.seam_positions.reserve(path_count);
  data.hops.reserve(event_count);

  for (std::size_t particle_index = 0; particle_index < path_count; ++particle_index) {
    const ContinuousPath &path = *path_at(particle_index);
    SiteId position = layout.encode_covering(path.start());
    data.seam_positions.push_back(position);
    for (const JumpEvent &event : path.events()) {
      const SiteId arrival = layout.shifted(position, static_cast<std::size_t>(event.axis),
                                            static_cast<Coord>(event.direction));
      data.hops.push_back(ContinuousHop{
          .time = normalize_event_time(event.time, model.beta()),
          .particle = static_cast<ParticleId>(particle_index),
          .departure = position,
          .arrival = arrival,
          .axis = event.axis,
          .direction = event.direction,
      });
      position = arrival;
    }
    if (position != layout.encode_covering(path.end())) {
      throw std::logic_error("continuous event replay does not reach the path endpoint");
    }
  }
  if (data.hops.size() != event_count) {
    throw std::logic_error("continuous event collection produced the wrong hop count");
  }

  std::ranges::stable_sort(data.hops, {}, &ContinuousHop::time);
  const std::size_t maximum_group_offsets =
      checked_add_size(event_count, 1, "continuous event-group extent exceeds size_t");
  require_vector_capacity(maximum_group_offsets, data.group_offsets.max_size(),
                          "continuous event-group extent exceeds vector capacity");
  data.group_offsets.reserve(maximum_group_offsets);
  data.group_offsets.push_back(0);
  std::size_t group_begin = 0;
  while (group_begin < data.hops.size()) {
    const double time = data.hops[group_begin].time;
    std::size_t group_end = group_begin + 1;
    while (group_end < data.hops.size() && data.hops[group_end].time == time) {
      ++group_end;
    }
    data.group_offsets.push_back(group_end);
    group_begin = group_end;
  }
  return data;
}

} // namespace

ContinuousEventSweepData build_continuous_event_sweep(const ContinuousConfiguration &configuration,
                                                      const TorusLayout &layout) {
  const Model &model = configuration.model();
  const auto worldlines = configuration.worldlines();
  return build_continuous_event_sweep_impl(
      model, worldlines.size(),
      [&worldlines](const std::size_t particle) { return &worldlines[particle]; }, layout);
}

ContinuousEventSweepData
build_continuous_event_sweep(const Model &model,
                             const std::span<const ContinuousPath *const> worldlines,
                             const TorusLayout &layout) {
  return build_continuous_event_sweep_impl(
      model, worldlines.size(),
      [worldlines](const std::size_t particle) { return worldlines[particle]; }, layout);
}

} // namespace qmc::detail
