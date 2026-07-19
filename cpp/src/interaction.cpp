#include "qmc/interaction.hpp"

#include "checked_math.hpp"
#include "interaction_detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace qmc {
namespace {

struct TimedEvent {
  double time;
  std::size_t particle;
  Axis axis;
  std::int8_t direction;
};

std::size_t site_index(const Site &site, const Model &model) {
  if (site.size() != model.dimension) {
    throw std::logic_error("site dimension does not match the model");
  }
  const auto linear_size = static_cast<std::size_t>(model.linear_size);
  std::size_t multiplier = 1;
  std::size_t index = 0;
  for (std::size_t axis = 0; axis < model.dimension; ++axis) {
    const auto coordinate = static_cast<std::size_t>(torus_mod(site[axis], model.linear_size));
    const auto contribution =
        detail::checked_product(coordinate, multiplier, "flat torus site index exceeds size_t");
    index = detail::checked_add_size(index, contribution, "flat torus site index exceeds size_t");
    if (axis + 1 < model.dimension) {
      multiplier = detail::checked_product(multiplier, linear_size,
                                           "flat torus site multiplier exceeds size_t");
    }
  }
  return index;
}

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

void ensure_finite_result(const double value, const char *description) {
  if (!std::isfinite(value)) {
    throw std::overflow_error(description);
  }
}

using OccupancyMap = std::unordered_map<std::size_t, std::uint64_t>;

struct OverlapWorkspace {
  std::vector<Site> positions;
  OccupancyMap occupancies;
  std::uint64_t pair_count = 0;
  std::size_t total_events = 0;
};

void add_pairs(std::uint64_t &pair_count, const std::uint64_t additional) {
  if (pair_count > std::numeric_limits<std::uint64_t>::max() - additional) {
    throw std::overflow_error("on-site pair count exceeds uint64 range");
  }
  pair_count += additional;
}

OverlapWorkspace initialize_workspace(const Model &model,
                                      const std::span<const ContinuousPath *const> worldlines) {
  OverlapWorkspace workspace;
  workspace.positions.reserve(worldlines.size());
  workspace.occupancies.reserve(worldlines.size());
  for (const ContinuousPath *path : worldlines) {
    if (path == nullptr) {
      throw std::logic_error("pair-overlap path view contains null");
    }
    path->validate(model.dimension);
    Site position = path->start;
    for (Coord &coordinate : position) {
      coordinate = torus_mod(coordinate, model.linear_size);
    }
    const std::size_t key = site_index(position, model);
    auto [entry, inserted] = workspace.occupancies.try_emplace(key, 0);
    static_cast<void>(inserted);
    add_pairs(workspace.pair_count, entry->second);
    ++entry->second;
    workspace.positions.push_back(std::move(position));
    workspace.total_events = detail::checked_add_size(workspace.total_events, path->events.size(),
                                                      "pair-overlap event count exceeds size_t");
  }
  return workspace;
}

std::vector<TimedEvent> collect_events(const Model &model,
                                       const std::span<const ContinuousPath *const> worldlines,
                                       const std::size_t total_events) {
  std::vector<TimedEvent> events;
  if (total_events > events.max_size()) {
    throw std::length_error("pair-overlap event count exceeds vector capacity");
  }
  events.reserve(total_events);
  for (std::size_t particle = 0; particle < worldlines.size(); ++particle) {
    for (const JumpEvent &event : worldlines[particle]->events) {
      events.push_back(TimedEvent{
          .time = normalize_event_time(event.time, model.beta),
          .particle = particle,
          .axis = event.axis,
          .direction = event.direction,
      });
    }
  }
  std::ranges::stable_sort(events, {}, &TimedEvent::time);
  return events;
}

void apply_event(const TimedEvent &event, const Model &model, OverlapWorkspace &workspace) {
  Site &position = workspace.positions[event.particle];
  const std::size_t old_key = site_index(position, model);
  auto old_entry = workspace.occupancies.find(old_key);
  if (old_entry == workspace.occupancies.end() || old_entry->second == 0) {
    throw std::logic_error("pair-overlap occupancy state is inconsistent");
  }
  workspace.pair_count -= old_entry->second - 1;
  --old_entry->second;
  if (old_entry->second == 0) {
    workspace.occupancies.erase(old_entry);
  }

  position[event.axis] =
      torus_mod(detail::checked_add(position[event.axis], static_cast<Coord>(event.direction),
                                    "physical event coordinate exceeds int64 range"),
                model.linear_size);
  const std::size_t new_key = site_index(position, model);
  auto [new_entry, inserted] = workspace.occupancies.try_emplace(new_key, 0);
  static_cast<void>(inserted);
  add_pairs(workspace.pair_count, new_entry->second);
  ++new_entry->second;
}

} // namespace

namespace detail {

double pair_overlap_time_for_paths(const Model &model,
                                   const std::span<const ContinuousPath *const> worldlines) {
  model.validate();
  if (model.beta <= 0.0) {
    throw std::invalid_argument("pair overlap requires positive beta");
  }
  if (worldlines.size() != model.particle_count) {
    throw std::logic_error("pair-overlap path count does not match particle_count");
  }
  static_cast<void>(model.volume());
  if (worldlines.size() < 2) {
    return 0.0;
  }
  OverlapWorkspace workspace = initialize_workspace(model, worldlines);
  const std::vector<TimedEvent> events = collect_events(model, worldlines, workspace.total_events);

  double overlap = 0.0;
  double previous_time = 0.0;
  std::size_t event_index = 0;
  while (event_index < events.size()) {
    const double event_time = events[event_index].time;
    overlap += (event_time - previous_time) * static_cast<double>(workspace.pair_count);
    ensure_finite_result(overlap, "pair-overlap integral overflowed");

    std::size_t group_end = event_index + 1;
    while (group_end < events.size() && events[group_end].time == event_time) {
      ++group_end;
    }
    for (std::size_t index = event_index; index < group_end; ++index) {
      apply_event(events[index], model, workspace);
    }
    previous_time = event_time;
    event_index = group_end;
  }

  overlap += (model.beta - previous_time) * static_cast<double>(workspace.pair_count);
  ensure_finite_result(overlap, "pair-overlap integral overflowed");
  if (overlap < 0.0) {
    throw std::logic_error("pair-overlap integral became negative");
  }
  return overlap;
}

} // namespace detail

double pair_overlap_time(const ContinuousConfiguration &configuration, const Model &model) {
  configuration.validate(model);
  std::vector<const ContinuousPath *> paths;
  paths.reserve(configuration.worldlines.size());
  for (const ContinuousPath &path : configuration.worldlines) {
    paths.push_back(&path);
  }
  return detail::pair_overlap_time_for_paths(model, paths);
}

double interaction_action(const ContinuousConfiguration &configuration,
                          const InteractingModel &model, const double chemical_potential) {
  model.validate();
  if (!std::isfinite(chemical_potential)) {
    throw std::invalid_argument("chemical potential must be finite");
  }
  const double action =
      (model.interaction * pair_overlap_time(configuration, model.free)) -
      (chemical_potential * static_cast<double>(model.free.particle_count) * model.free.beta);
  ensure_finite_result(action, "interaction action overflowed");
  return action;
}

double kinetic_energy_estimator(const ContinuousConfiguration &configuration, const Model &model) {
  configuration.validate(model);
  if (model.beta <= 0.0) {
    throw std::invalid_argument("kinetic energy estimator requires positive beta");
  }
  return -static_cast<double>(configuration.event_count()) / model.beta;
}

double interaction_energy_estimator(const ContinuousConfiguration &configuration,
                                    const InteractingModel &model) {
  model.validate();
  const double energy =
      model.interaction * pair_overlap_time(configuration, model.free) / model.free.beta;
  ensure_finite_result(energy, "interaction energy estimator overflowed");
  return energy;
}

double total_energy_estimator(const ContinuousConfiguration &configuration,
                              const InteractingModel &model) {
  return kinetic_energy_estimator(configuration, model.free) +
         interaction_energy_estimator(configuration, model);
}

double double_occupancy_per_site(const ContinuousConfiguration &configuration, const Model &model) {
  configuration.validate(model);
  if (model.beta <= 0.0) {
    throw std::invalid_argument("double occupancy requires positive beta");
  }
  return pair_overlap_time(configuration, model) /
         (model.beta * static_cast<double>(model.volume()));
}

} // namespace qmc
