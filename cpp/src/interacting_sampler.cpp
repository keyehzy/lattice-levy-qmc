#include "qmc/interacting_sampler.hpp"

#include "continuous_detail.hpp"
#include "interaction_detail.hpp"
#include "qmc/interaction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qmc {
namespace {

std::size_t move_index(const MoveKind move) {
  switch (move) {
  case MoveKind::SegmentMove:
    return 0;
  case MoveKind::CycleMove:
    return 1;
  case MoveKind::GlobalMove:
    return 2;
  }
  throw std::invalid_argument("unknown move kind");
}

void ensure_counter_capacity(const std::uint64_t counter) {
  if (counter == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("move statistics counter overflowed");
  }
}

void increment_counter(std::uint64_t &counter) {
  ensure_counter_capacity(counter);
  ++counter;
}

double checked_action(const double interaction, const double overlap) {
  const double action = interaction * overlap;
  if (!std::isfinite(action)) {
    throw std::overflow_error("interaction action overflowed");
  }
  return action;
}

} // namespace

std::string_view move_name(const MoveKind move) noexcept {
  switch (move) {
  case MoveKind::SegmentMove:
    return "segment";
  case MoveKind::CycleMove:
    return "cycle";
  case MoveKind::GlobalMove:
    return "global";
  }
  return "unknown";
}

std::optional<double> MoveStatistics::acceptance() const noexcept {
  if (attempts == 0) {
    return std::nullopt;
  }
  return static_cast<double>(accepts) / static_cast<double>(attempts);
}

InteractingSampler::InteractingSampler(InteractingModel model, NumericalOptions numerical,
                                       const std::uint64_t seed)
    : model_(model), numerical_(numerical), random_(seed) {
  model_.validate();
  numerical_.validate();
  free_table_ = canonical_table(model_.free);
  state_ = sample_ideal_continuous_configuration(model_.free, free_table_, random_, numerical_);
  pair_overlap_ = pair_overlap_time(state_, model_.free);
  action_ = checked_action(model_.interaction, pair_overlap_);
}

const MoveStatistics &InteractingSampler::statistics(const MoveKind move) const {
  return statistics_[move_index(move)];
}

bool InteractingSampler::metropolis_accept(const double delta_action) {
  if (std::isnan(delta_action)) {
    throw std::runtime_error("Metropolis action difference is NaN");
  }
  if (delta_action <= 0.0) {
    return true;
  }
  if (delta_action == std::numeric_limits<double>::infinity()) {
    return false;
  }
  return std::log(random_.uniform_open()) < -delta_action;
}

bool InteractingSampler::try_path_replacements(std::vector<LabeledPath> replacements,
                                               const MoveKind move) {
  std::vector<const ContinuousPath *> path_view;
  path_view.reserve(state_.worldlines.size());
  for (const ContinuousPath &path : state_.worldlines) {
    path_view.push_back(&path);
  }
  std::vector<bool> replaced(state_.worldlines.size(), false);
  for (LabeledPath &replacement : replacements) {
    const auto label = static_cast<std::size_t>(replacement.first);
    if (label >= state_.worldlines.size() || replaced[label]) {
      throw std::logic_error("path replacements contain an invalid or duplicate label");
    }
    replacement.second.validate(model_.free.dimension);
    replaced[label] = true;
    path_view[label] = &replacement.second;
  }

  MoveStatistics &move_statistics = statistics_[move_index(move)];
  increment_counter(move_statistics.attempts);
  const double new_overlap = detail::pair_overlap_time_for_paths(model_.free, path_view);
  const double new_action = checked_action(model_.interaction, new_overlap);
  const bool accepted = metropolis_accept(new_action - action_);
  if (!accepted) {
    return false;
  }

  ensure_counter_capacity(move_statistics.accepts);
  for (LabeledPath &replacement : replacements) {
    std::swap(state_.worldlines[replacement.first], replacement.second);
  }
  pair_overlap_ = new_overlap;
  action_ = new_action;
  increment_counter(move_statistics.accepts);
  return true;
}

bool InteractingSampler::segment_update(const std::optional<ParticleId> particle,
                                        const std::optional<std::pair<double, double>> interval,
                                        const double fraction) {
  if (model_.free.particle_count == 0) {
    return true;
  }
  const ParticleId label = particle.has_value()
                               ? *particle
                               : static_cast<ParticleId>(random_.uniform_index(
                                     static_cast<std::uint64_t>(model_.free.particle_count)));
  if (static_cast<std::size_t>(label) >= model_.free.particle_count) {
    throw std::invalid_argument("particle label is out of range");
  }

  double tau0 = 0.0;
  double tau1 = 0.0;
  if (interval.has_value()) {
    tau0 = interval->first;
    tau1 = interval->second;
  } else {
    if (!std::isfinite(fraction) || fraction <= 0.0 || fraction > 1.0) {
      throw std::invalid_argument("segment fraction must lie in (0, 1]");
    }
    const double duration = fraction * model_.free.beta;
    if (duration == model_.free.beta) {
      tau0 = 0.0;
    } else {
      tau0 = random_.uniform_unit() * (model_.free.beta - duration);
    }
    tau1 = std::min(model_.free.beta, tau0 + duration);
  }

  ContinuousPath proposal = resample_path_interval(state_.worldlines[label], tau0, tau1,
                                                   model_.free.hopping, random_, numerical_);
  std::vector<LabeledPath> replacements;
  replacements.emplace_back(label, std::move(proposal));
  return try_path_replacements(std::move(replacements), MoveKind::SegmentMove);
}

bool InteractingSampler::whole_worldline_update(const std::optional<ParticleId> particle) {
  return segment_update(particle,
                        std::optional<std::pair<double, double>>{{0.0, model_.free.beta}});
}

bool InteractingSampler::cycle_update(const std::optional<std::size_t> cycle_index) {
  if (state_.cycles.empty()) {
    return true;
  }
  const std::size_t selected = cycle_index.has_value()
                                   ? *cycle_index
                                   : static_cast<std::size_t>(random_.uniform_index(
                                         static_cast<std::uint64_t>(state_.cycles.size())));
  if (selected >= state_.cycles.size()) {
    throw std::invalid_argument("cycle index is out of range");
  }
  const Cycle &cycle = state_.cycles[selected];
  auto proposed_paths = detail::sample_paths_for_cycle(cycle, model_.free, random_, numerical_);
  std::vector<LabeledPath> replacements;
  replacements.reserve(cycle.size());
  for (std::size_t index = 0; index < cycle.size(); ++index) {
    replacements.emplace_back(cycle[index], std::move(proposed_paths[index]));
  }
  return try_path_replacements(std::move(replacements), MoveKind::CycleMove);
}

bool InteractingSampler::global_update() {
  ContinuousConfiguration proposal =
      sample_ideal_continuous_configuration(model_.free, free_table_, random_, numerical_);
  const double new_overlap = pair_overlap_time(proposal, model_.free);
  const double new_action = checked_action(model_.interaction, new_overlap);
  MoveStatistics &move_statistics = statistics_[move_index(MoveKind::GlobalMove)];
  increment_counter(move_statistics.attempts);
  const bool accepted = metropolis_accept(new_action - action_);
  if (!accepted) {
    return false;
  }
  ensure_counter_capacity(move_statistics.accepts);
  std::swap(state_, proposal);
  pair_overlap_ = new_overlap;
  action_ = new_action;
  increment_counter(move_statistics.accepts);
  return true;
}

void InteractingSampler::sweep(const SweepOptions &options) {
  const std::size_t segment_count = options.segment_updates.value_or(model_.free.particle_count);
  for (std::size_t update = 0; update < segment_count; ++update) {
    static_cast<void>(segment_update(std::nullopt, std::nullopt, options.segment_fraction));
  }
  for (std::size_t update = 0; update < options.cycle_updates; ++update) {
    static_cast<void>(cycle_update());
  }
  for (std::size_t update = 0; update < options.global_updates; ++update) {
    static_cast<void>(global_update());
  }
}

InteractingObservables InteractingSampler::observables() const {
  const auto event_count = state_.event_count();
  const double kinetic = -static_cast<double>(event_count) / model_.free.beta;
  const double interaction = model_.interaction * pair_overlap_ / model_.free.beta;
  return InteractingObservables{
      .action = action_,
      .pair_overlap_time = pair_overlap_,
      .double_occupancy_per_site =
          pair_overlap_ / (model_.free.beta * static_cast<double>(model_.free.volume())),
      .kinetic_energy = kinetic,
      .interaction_energy = interaction,
      .total_energy = kinetic + interaction,
      .event_count = event_count,
      .winding = state_.total_winding(model_.free),
      .cycle_lengths = state_.cycle_lengths(),
  };
}

std::vector<InteractingObservables> InteractingSampler::run(const std::size_t sample_count,
                                                            const RunOptions &options) {
  if (options.thin == 0) {
    throw std::invalid_argument("run thinning interval must be positive");
  }
  for (std::size_t sweep_index = 0; sweep_index < options.burn_in; ++sweep_index) {
    sweep(options.sweep);
  }

  std::vector<InteractingObservables> output;
  if (sample_count > output.max_size()) {
    throw std::length_error("sample count exceeds vector capacity");
  }
  output.reserve(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    for (std::size_t thin_index = 0; thin_index < options.thin; ++thin_index) {
      sweep(options.sweep);
    }
    output.push_back(observables());
  }
  return output;
}

} // namespace qmc
