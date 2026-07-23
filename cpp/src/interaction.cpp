#include "qmc/interaction.hpp"

#include "continuous_event_sweep.hpp"
#include "interaction_detail.hpp"
#include "qmc/torus_layout.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace qmc {
namespace {

void ensure_finite_result(const double value, const char *description) {
  if (!std::isfinite(value)) {
    throw std::overflow_error(description);
  }
}

using OccupancyMap = std::unordered_map<SiteId, std::uint64_t, SiteIdHash>;

struct OverlapWorkspace {
  std::vector<SiteId> positions;
  OccupancyMap occupancies;
  std::uint64_t pair_count = 0;
};

void add_pairs(std::uint64_t &pair_count, const std::uint64_t additional) {
  if (pair_count > std::numeric_limits<std::uint64_t>::max() - additional) {
    throw std::overflow_error("on-site pair count exceeds uint64 range");
  }
  pair_count += additional;
}

OverlapWorkspace initialize_workspace(const std::span<const SiteId> seam_positions) {
  OverlapWorkspace workspace;
  workspace.positions.reserve(seam_positions.size());
  workspace.occupancies.reserve(seam_positions.size());
  for (const SiteId key : seam_positions) {
    auto [entry, inserted] = workspace.occupancies.try_emplace(key, 0);
    static_cast<void>(inserted);
    add_pairs(workspace.pair_count, entry->second);
    ++entry->second;
    workspace.positions.push_back(key);
  }
  return workspace;
}

void apply_hop(const ContinuousHop &hop, OverlapWorkspace &workspace) {
  const auto particle = static_cast<std::size_t>(hop.particle);
  if (particle >= workspace.positions.size()) {
    throw std::logic_error("pair-overlap hop contains an invalid particle label");
  }
  SiteId &position = workspace.positions[particle];
  if (position != hop.departure) {
    throw std::logic_error("pair-overlap replay disagrees with the shared event sweep");
  }
  const SiteId old_key = position;
  auto old_entry = workspace.occupancies.find(old_key);
  if (old_entry == workspace.occupancies.end() || old_entry->second == 0) {
    throw std::logic_error("pair-overlap occupancy state is inconsistent");
  }
  workspace.pair_count -= old_entry->second - 1;
  --old_entry->second;
  if (old_entry->second == 0) {
    workspace.occupancies.erase(old_entry);
  }

  position = hop.arrival;
  const SiteId new_key = position;
  auto [new_entry, inserted] = workspace.occupancies.try_emplace(new_key, 0);
  static_cast<void>(inserted);
  add_pairs(workspace.pair_count, new_entry->second);
  ++new_entry->second;
}

} // namespace

namespace detail {

void validate_interaction_model_provenance(const ContinuousConfiguration &configuration,
                                           const InteractingModel &model) {
  model.validate();
  if (configuration.model() != model.free) {
    throw std::invalid_argument(
        "continuous configuration provenance does not match the interacting model");
  }
}

double pair_overlap_time_for_paths(const Model &model,
                                   const std::span<const ContinuousPath *const> worldlines) {
  if (model.beta() <= 0.0) {
    throw std::invalid_argument("pair overlap requires positive beta");
  }
  if (worldlines.size() != model.particle_count()) {
    throw std::logic_error("pair-overlap path count does not match particle_count");
  }
  const TorusLayout layout(model.linear_size(), model.dimension());
  if (worldlines.size() < 2) {
    return 0.0;
  }
  const ContinuousEventSweepData sweep = build_continuous_event_sweep(model, worldlines, layout);
  OverlapWorkspace workspace = initialize_workspace(sweep.seam_positions);

  double overlap = 0.0;
  double previous_time = 0.0;
  for (std::size_t group = 0; group + 1 < sweep.group_offsets.size(); ++group) {
    const std::size_t group_begin = sweep.group_offsets[group];
    const std::size_t group_end = sweep.group_offsets[group + 1];
    const double event_time = sweep.hops[group_begin].time;
    overlap += (event_time - previous_time) * static_cast<double>(workspace.pair_count);
    ensure_finite_result(overlap, "pair-overlap integral overflowed");

    for (std::size_t index = group_begin; index < group_end; ++index) {
      apply_hop(sweep.hops[index], workspace);
    }
    previous_time = event_time;
  }

  overlap += (model.beta() - previous_time) * static_cast<double>(workspace.pair_count);
  ensure_finite_result(overlap, "pair-overlap integral overflowed");
  if (overlap < 0.0) {
    throw std::logic_error("pair-overlap integral became negative");
  }
  return overlap;
}

InteractionMeasurement
interaction_measurement_from_validated_overlap(const ContinuousConfiguration &configuration,
                                               const InteractingModel &model,
                                               const double pair_overlap) {
  if (!std::isfinite(pair_overlap) || pair_overlap < 0.0) {
    throw std::logic_error("interaction measurement requires a finite nonnegative pair overlap");
  }

  const auto event_count = configuration.event_count();
  const double beta = model.free.beta();
  const double action = model.interaction * pair_overlap;
  const double kinetic_energy = -static_cast<double>(event_count) / beta;
  const double interaction_energy = action / beta;
  const double total_energy = kinetic_energy + interaction_energy;
  const double double_occupancy = pair_overlap / (beta * static_cast<double>(model.free.volume()));
  ensure_finite_result(action, "interaction action overflowed");
  ensure_finite_result(kinetic_energy, "kinetic energy estimator overflowed");
  ensure_finite_result(interaction_energy, "interaction energy estimator overflowed");
  ensure_finite_result(total_energy, "total energy estimator overflowed");
  ensure_finite_result(double_occupancy, "double occupancy estimator overflowed");

  return InteractionMeasurement{
      .action = action,
      .pair_overlap_time = pair_overlap,
      .double_occupancy_per_site = double_occupancy,
      .kinetic_energy = kinetic_energy,
      .interaction_energy = interaction_energy,
      .total_energy = total_energy,
      .event_count = event_count,
  };
}

} // namespace detail

double pair_overlap_time(const ContinuousConfiguration &configuration) {
  std::vector<const ContinuousPath *> paths;
  paths.reserve(configuration.worldlines().size());
  for (const ContinuousPath &path : configuration.worldlines()) {
    paths.push_back(&path);
  }
  return detail::pair_overlap_time_for_paths(configuration.model(), paths);
}

InteractionMeasurement measure_interaction(const ContinuousConfiguration &configuration,
                                           const InteractingModel &model) {
  detail::validate_interaction_model_provenance(configuration, model);
  return detail::interaction_measurement_from_validated_overlap(configuration, model,
                                                                pair_overlap_time(configuration));
}

double interaction_action(const ContinuousConfiguration &configuration,
                          const InteractingModel &model, const double chemical_potential) {
  detail::validate_interaction_model_provenance(configuration, model);
  if (!std::isfinite(chemical_potential)) {
    throw std::invalid_argument("chemical potential must be finite");
  }
  const double action =
      (model.interaction * pair_overlap_time(configuration)) -
      (chemical_potential * static_cast<double>(model.free.particle_count()) * model.free.beta());
  ensure_finite_result(action, "interaction action overflowed");
  return action;
}

double kinetic_energy_estimator(const ContinuousConfiguration &configuration) {
  return -static_cast<double>(configuration.event_count()) / configuration.model().beta();
}

double interaction_energy_estimator(const ContinuousConfiguration &configuration,
                                    const InteractingModel &model) {
  detail::validate_interaction_model_provenance(configuration, model);
  const double energy = model.interaction * pair_overlap_time(configuration) / model.free.beta();
  ensure_finite_result(energy, "interaction energy estimator overflowed");
  return energy;
}

double total_energy_estimator(const ContinuousConfiguration &configuration,
                              const InteractingModel &model) {
  return kinetic_energy_estimator(configuration) +
         interaction_energy_estimator(configuration, model);
}

double double_occupancy_per_site(const ContinuousConfiguration &configuration) {
  const Model &model = configuration.model();
  return pair_overlap_time(configuration) / (model.beta() * static_cast<double>(model.volume()));
}

} // namespace qmc
