#include "qmc/interacting_sampler.hpp"

#include "checked_math.hpp"
#include "continuous_detail.hpp"
#include "interaction_detail.hpp"
#include "occupancy_index.hpp"
#include "qmc/interaction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
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
  case MoveKind::StitchMove:
    return 3;
  case MoveKind::TimeShiftMove:
    return 4;
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

struct SiteHash {
  std::size_t operator()(const Site &site) const noexcept {
    std::size_t result = site.size();
    for (const Coord coordinate : site) {
      const auto value = static_cast<std::uint64_t>(coordinate);
      result ^=
          static_cast<std::size_t>(value + 0x9e3779b97f4a7c15ULL + (result << 6U) + (result >> 2U));
    }
    return result;
  }
};

using PartnerBuckets = std::unordered_map<Site, std::vector<ParticleId>, SiteHash>;

bool within_torus_radius(const Site &left, const Site &right, const Coord linear_size,
                         const std::size_t radius) {
  if (left.size() != right.size()) {
    throw std::logic_error("stitch bucket site dimension mismatch");
  }
  for (std::size_t axis = 0; axis < left.size(); ++axis) {
    const Coord direct =
        left[axis] >= right[axis] ? left[axis] - right[axis] : right[axis] - left[axis];
    const Coord periodic = linear_size - direct;
    const Coord distance = std::min(direct, periodic);
    if (std::cmp_greater(distance, radius)) {
      return false;
    }
  }
  return true;
}

PartnerBuckets build_partner_buckets(const std::vector<Site> &positions) {
  PartnerBuckets buckets;
  buckets.reserve(positions.size());
  for (std::size_t label = 0; label < positions.size(); ++label) {
    buckets[positions[label]].push_back(static_cast<ParticleId>(label));
  }
  return buckets;
}

Coord shifted_torus_coordinate(const Coord center, const Coord offset, const Coord linear_size) {
  if (offset >= 0) {
    const Coord distance_to_wrap = linear_size - center;
    return offset < distance_to_wrap ? center + offset : offset - distance_to_wrap;
  }
  const Coord magnitude = -offset;
  return magnitude <= center ? center - magnitude : linear_size - (magnitude - center);
}

void collect_neighbor_candidates(const std::size_t axis, Site &site, const Site &center,
                                 const Coord radius, const Coord linear_size,
                                 const PartnerBuckets &buckets, const ParticleId particle,
                                 std::vector<ParticleId> &candidates) {
  if (axis == site.size()) {
    const auto bucket = buckets.find(site);
    if (bucket == buckets.end()) {
      return;
    }
    for (const ParticleId label : bucket->second) {
      if (label != particle) {
        candidates.push_back(label);
      }
    }
    return;
  }

  for (Coord offset = -radius;; ++offset) {
    site[axis] = shifted_torus_coordinate(center[axis], offset, linear_size);
    collect_neighbor_candidates(axis + 1, site, center, radius, linear_size, buckets, particle,
                                candidates);
    if (offset == radius) {
      break;
    }
  }
}

bool neighborhood_is_small(const Model &model, const std::size_t locality_radius,
                           const std::size_t occupied_sites) {
  if (locality_radius > static_cast<std::size_t>(std::numeric_limits<Coord>::max())) {
    return false;
  }
  const auto radius = static_cast<Coord>(locality_radius);
  if (radius >= (model.linear_size / 2) + (model.linear_size % 2)) {
    return false;
  }
  const std::size_t width = (2 * locality_radius) + 1;
  const std::size_t limit = occupied_sites > (std::numeric_limits<std::size_t>::max() - 1) / 4
                                ? std::numeric_limits<std::size_t>::max()
                                : (4 * occupied_sites) + 1;
  std::size_t volume = 1;
  for (std::size_t axis = 0; axis < model.dimension; ++axis) {
    if (volume > limit / width) {
      return false;
    }
    volume *= width;
  }
  return true;
}

std::vector<ParticleId> same_site_candidates(const ParticleId particle, const Site &position,
                                             const PartnerBuckets &buckets) {
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

std::vector<ParticleId> scan_neighbor_candidates(const ParticleId particle, const Site &position,
                                                 const PartnerBuckets &buckets, const Model &model,
                                                 const std::size_t locality_radius) {
  std::vector<ParticleId> candidates;
  for (const auto &[site, labels] : buckets) {
    if (!within_torus_radius(position, site, model.linear_size, locality_radius)) {
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
                                                const std::vector<Site> &positions,
                                                const PartnerBuckets &buckets, const Model &model,
                                                const std::size_t locality_radius) {
  if (locality_radius == 0) {
    return same_site_candidates(particle, positions[particle], buckets);
  }
  if (!neighborhood_is_small(model, locality_radius, buckets.size())) {
    return scan_neighbor_candidates(particle, positions[particle], buckets, model, locality_radius);
  }

  std::vector<ParticleId> candidates;
  Site neighbor(model.dimension, 0);
  collect_neighbor_candidates(0, neighbor, positions[particle], static_cast<Coord>(locality_radius),
                              model.linear_size, buckets, particle, candidates);
  return candidates;
}

ParticleId select_stitch_partner(const ParticleId particle, const std::vector<Site> &positions,
                                 const PartnerBuckets &buckets, const Model &model,
                                 const std::size_t locality_radius,
                                 const double global_partner_probability, Random &random) {
  if (!std::isfinite(global_partner_probability) || global_partner_probability < 0.0 ||
      global_partner_probability > 1.0) {
    throw std::invalid_argument("global partner probability must lie in [0, 1]");
  }
  if (model.particle_count < 2 || static_cast<std::size_t>(particle) >= model.particle_count) {
    throw std::invalid_argument("cannot select a stitch partner for this particle");
  }

  const bool use_global = random.uniform_unit() < global_partner_probability;
  std::vector<ParticleId> candidates;
  if (!use_global) {
    candidates = local_stitch_candidates(particle, positions, buckets, model, locality_radius);
  }
  if (!candidates.empty()) {
    return candidates[random.uniform_index(static_cast<std::uint64_t>(candidates.size()))];
  }

  const auto draw = static_cast<std::size_t>(
      random.uniform_index(static_cast<std::uint64_t>(model.particle_count - 1)));
  const auto particle_index = static_cast<std::size_t>(particle);
  return static_cast<ParticleId>(draw < particle_index ? draw : draw + 1);
}

ContinuousPath splice_path_interval(const ContinuousPath &prefix_path,
                                    const ContinuousPath &suffix_path, const ContinuousPath &bridge,
                                    const double tau0, const double tau1, const Coord linear_size) {
  if (prefix_path.duration != suffix_path.duration || bridge.duration != tau1 - tau0) {
    throw std::invalid_argument("stitch path durations do not match the splice window");
  }
  const Site left = prefix_path.position_at(tau0);
  const Site right = suffix_path.position_at(tau1);
  if (bridge.start != left || bridge.end.size() != right.size()) {
    throw std::invalid_argument("stitch bridge does not start at the prefix cut");
  }
  for (std::size_t axis = 0; axis < right.size(); ++axis) {
    if (torus_mod(bridge.end[axis], linear_size) != torus_mod(right[axis], linear_size)) {
      throw std::invalid_argument("stitch bridge does not end at the suffix torus site");
    }
  }

  std::vector<JumpEvent> events;
  events.reserve(detail::checked_add_size(
      detail::checked_add_size(prefix_path.events.size(), bridge.events.size(),
                               "stitched event count exceeds size_t"),
      suffix_path.events.size(), "stitched event count exceeds size_t"));
  for (const JumpEvent &event : prefix_path.events) {
    if (event.time <= tau0) {
      events.push_back(event);
    }
  }
  for (const JumpEvent &event : bridge.events) {
    events.push_back(JumpEvent{
        .time = tau0 + event.time,
        .axis = event.axis,
        .direction = event.direction,
    });
  }
  for (const JumpEvent &event : suffix_path.events) {
    if (event.time > tau1) {
      events.push_back(event);
    }
  }

  Site end(bridge.end.size());
  for (std::size_t axis = 0; axis < end.size(); ++axis) {
    const Coord suffix_displacement = detail::checked_subtract(
        suffix_path.end[axis], right[axis], "stitch suffix displacement exceeds int64 range");
    end[axis] = detail::checked_add(bridge.end[axis], suffix_displacement,
                                    "stitched path endpoint exceeds int64 range");
  }
  ContinuousPath result{
      .duration = prefix_path.duration,
      .start = prefix_path.start,
      .end = std::move(end),
      .events = std::move(events),
  };
  result.validate(prefix_path.start.size());
  return result;
}

} // namespace

std::string_view move_name(const MoveKind move) noexcept {
  switch (move) {
  case MoveKind::SegmentMove:
    return "segment";
  case MoveKind::CycleMove:
    return "cycle";
  case MoveKind::StitchMove:
    return "stitch";
  case MoveKind::TimeShiftMove:
    return "time_shift";
  case MoveKind::GlobalMove:
    return "global";
  }
  return "unknown";
}

std::optional<double> MoveStatistics::topology_change_rate() const noexcept {
  if (attempts == 0) {
    return std::nullopt;
  }
  return static_cast<double>(topology_changes) / static_cast<double>(attempts);
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
  rebuild_occupancy_index();
}

InteractingSampler::~InteractingSampler() = default;
InteractingSampler::InteractingSampler(const InteractingSampler &other)
    : model_(other.model_), numerical_(other.numerical_), random_(other.random_),
      free_table_(other.free_table_), state_(other.state_), pair_overlap_(other.pair_overlap_),
      action_(other.action_), statistics_(other.statistics_) {
  rebuild_occupancy_index();
}

InteractingSampler &InteractingSampler::operator=(const InteractingSampler &other) {
  if (this != &other) {
    InteractingSampler copy(other);
    *this = std::move(copy);
  }
  return *this;
}

InteractingSampler::InteractingSampler(InteractingSampler &&) noexcept = default;
InteractingSampler &InteractingSampler::operator=(InteractingSampler &&) noexcept = default;

void InteractingSampler::rebuild_occupancy_index() {
  auto rebuilt = std::make_unique<detail::OccupancyIndex>(model_.free);
  rebuilt->rebuild(state_.worldlines);
  occupancy_index_ = std::move(rebuilt);
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
  std::ranges::sort(replacements, {}, &LabeledPath::first);
  std::vector<bool> replaced(state_.worldlines.size(), false);
  std::vector<const ContinuousPath *> old_paths;
  std::vector<const ContinuousPath *> new_paths;
  old_paths.reserve(replacements.size());
  new_paths.reserve(replacements.size());
  for (LabeledPath &replacement : replacements) {
    const auto label = static_cast<std::size_t>(replacement.first);
    if (label >= state_.worldlines.size() || replaced[label]) {
      throw std::logic_error("path replacements contain an invalid or duplicate label");
    }
    replacement.second.validate(model_.free.dimension);
    replaced[label] = true;
    old_paths.push_back(&state_.worldlines[label]);
    new_paths.push_back(&replacement.second);
  }

  MoveStatistics &move_statistics = statistics_[move_index(move)];
  increment_counter(move_statistics.attempts);
  const double new_overlap = occupancy_index_->replace_paths(old_paths, new_paths, pair_overlap_);
  double new_action = 0.0;
  bool accepted = false;
  try {
    new_action = checked_action(model_.interaction, new_overlap);
    accepted = metropolis_accept(new_action - action_);
  } catch (...) {
    occupancy_index_->rollback_replacement(old_paths, new_paths);
    throw;
  }
  if (!accepted) {
    occupancy_index_->rollback_replacement(old_paths, new_paths);
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

InteractingSampler::StitchProposal
InteractingSampler::sample_stitch_pair_proposal(const ParticleId particle, const ParticleId partner,
                                                const double tau0, const double tau1) {
  if (particle == partner || static_cast<std::size_t>(particle) >= state_.worldlines.size() ||
      static_cast<std::size_t>(partner) >= state_.worldlines.size()) {
    throw std::invalid_argument("stitch particles must be distinct valid labels");
  }
  const ContinuousPath &old_i = state_.worldlines[particle];
  const ContinuousPath &old_j = state_.worldlines[partner];
  const double duration = tau1 - tau0;
  const Site left_i = old_i.position_at(tau0);
  const Site left_j = old_j.position_at(tau0);
  const Site right_i = old_i.position_at(tau1);
  const Site right_j = old_j.position_at(tau1);

  const double log_identity =
      log_torus_kernel_scaled(left_i, right_i, duration, model_.free.linear_size,
                              model_.free.hopping, numerical_) +
      log_torus_kernel_scaled(left_j, right_j, duration, model_.free.linear_size,
                              model_.free.hopping, numerical_);
  const double log_exchange =
      log_torus_kernel_scaled(left_i, right_j, duration, model_.free.linear_size,
                              model_.free.hopping, numerical_) +
      log_torus_kernel_scaled(left_j, right_i, duration, model_.free.linear_size,
                              model_.free.hopping, numerical_);
  if (!std::isfinite(log_identity) && !std::isfinite(log_exchange)) {
    throw std::runtime_error("both stitch matchings have zero free weight");
  }

  bool exchanged = false;
  if (!std::isfinite(log_identity)) {
    exchanged = true;
  } else if (std::isfinite(log_exchange)) {
    const std::array<double, 2> log_weights{log_identity, log_exchange};
    const double log_normalizer = log_sum_exp(log_weights);
    exchanged = std::log(random_.uniform_open()) < log_exchange - log_normalizer;
  }

  const ContinuousPath &suffix_i = exchanged ? old_j : old_i;
  const ContinuousPath &suffix_j = exchanged ? old_i : old_j;
  const Site &target_i = exchanged ? right_j : right_i;
  const Site &target_j = exchanged ? right_i : right_j;
  const ContinuousPath bridge_i =
      sample_continuous_bridge_torus(left_i, target_i, duration, model_.free.linear_size,
                                     model_.free.hopping, random_, numerical_);
  const ContinuousPath bridge_j =
      sample_continuous_bridge_torus(left_j, target_j, duration, model_.free.linear_size,
                                     model_.free.hopping, random_, numerical_);

  StitchProposal proposal;
  proposal.replacements.emplace_back(particle, splice_path_interval(old_i, suffix_i, bridge_i, tau0,
                                                                    tau1, model_.free.linear_size));
  proposal.replacements.emplace_back(partner, splice_path_interval(old_j, suffix_j, bridge_j, tau0,
                                                                   tau1, model_.free.linear_size));
  proposal.permutation = state_.permutation;
  if (exchanged) {
    std::swap(proposal.permutation[particle], proposal.permutation[partner]);
  }
  proposal.exchanged = exchanged;
  return proposal;
}

bool InteractingSampler::try_stitch_pair(const ParticleId particle, const ParticleId partner,
                                         const double tau0, const double tau1) {
  MoveStatistics &move_statistics = statistics_[move_index(MoveKind::StitchMove)];
  increment_counter(move_statistics.attempts);
  StitchProposal proposal = sample_stitch_pair_proposal(particle, partner, tau0, tau1);
  std::ranges::sort(proposal.replacements, {}, &LabeledPath::first);
  std::vector<Cycle> proposed_cycles = cycles_from_permutation(proposal.permutation);

  std::vector<const ContinuousPath *> old_paths;
  std::vector<const ContinuousPath *> new_paths;
  old_paths.reserve(proposal.replacements.size());
  new_paths.reserve(proposal.replacements.size());
  for (LabeledPath &replacement : proposal.replacements) {
    old_paths.push_back(&state_.worldlines[replacement.first]);
    new_paths.push_back(&replacement.second);
  }
  const double new_overlap = occupancy_index_->replace_paths(old_paths, new_paths, pair_overlap_);
  double new_action = 0.0;
  bool accepted = false;
  try {
    new_action = checked_action(model_.interaction, new_overlap);
    accepted = metropolis_accept(new_action - action_);
  } catch (...) {
    occupancy_index_->rollback_replacement(old_paths, new_paths);
    throw;
  }
  if (!accepted) {
    occupancy_index_->rollback_replacement(old_paths, new_paths);
    return false;
  }

  ensure_counter_capacity(move_statistics.accepts);
  if (proposal.exchanged) {
    ensure_counter_capacity(move_statistics.topology_changes);
  }
  for (LabeledPath &replacement : proposal.replacements) {
    std::swap(state_.worldlines[replacement.first], replacement.second);
  }
  state_.permutation = std::move(proposal.permutation);
  state_.cycles = std::move(proposed_cycles);
  pair_overlap_ = new_overlap;
  action_ = new_action;
  increment_counter(move_statistics.accepts);
  if (proposal.exchanged) {
    increment_counter(move_statistics.topology_changes);
  }
  return true;
}

bool InteractingSampler::stitch_update(const std::optional<ParticleId> particle,
                                       const std::optional<ParticleId> partner,
                                       const std::optional<std::pair<double, double>> interval,
                                       const double fraction, const std::size_t locality_radius,
                                       const double global_partner_probability) {
  if (model_.free.particle_count < 2) {
    return true;
  }
  const ParticleId selected = particle.has_value()
                                  ? *particle
                                  : static_cast<ParticleId>(random_.uniform_index(
                                        static_cast<std::uint64_t>(model_.free.particle_count)));
  if (static_cast<std::size_t>(selected) >= model_.free.particle_count) {
    throw std::invalid_argument("particle label is out of range");
  }

  double tau0 = 0.0;
  double tau1 = 0.0;
  if (interval.has_value()) {
    tau0 = interval->first;
    tau1 = interval->second;
  } else {
    if (!std::isfinite(fraction) || fraction <= 0.0 || fraction > 1.0) {
      throw std::invalid_argument("stitch fraction must lie in (0, 1]");
    }
    const double duration = fraction * model_.free.beta;
    tau0 =
        duration == model_.free.beta ? 0.0 : random_.uniform_unit() * (model_.free.beta - duration);
    tau1 = std::min(model_.free.beta, tau0 + duration);
  }
  if (!std::isfinite(tau0) || !std::isfinite(tau1) || tau0 < 0.0 || tau1 > model_.free.beta ||
      tau1 <= tau0) {
    throw std::invalid_argument("require 0 <= stitch tau0 < tau1 <= beta");
  }

  ParticleId selected_partner = 0;
  if (partner.has_value()) {
    selected_partner = *partner;
  } else {
    const std::vector<Site> positions = state_.positions_at(tau0, model_.free);
    const PartnerBuckets buckets = build_partner_buckets(positions);
    selected_partner = select_stitch_partner(selected, positions, buckets, model_.free,
                                             locality_radius, global_partner_probability, random_);
  }
  if (static_cast<std::size_t>(selected_partner) >= model_.free.particle_count ||
      selected_partner == selected) {
    throw std::invalid_argument("partner must be a distinct valid particle label");
  }
  return try_stitch_pair(selected, selected_partner, tau0, tau1);
}

void InteractingSampler::stitch_sweep(const std::optional<std::size_t> updates,
                                      const double fraction,
                                      const std::optional<double> requested_tau0,
                                      const std::size_t locality_radius,
                                      const double global_partner_probability) {
  if (model_.free.particle_count < 2) {
    return;
  }
  if (!std::isfinite(fraction) || fraction <= 0.0 || fraction > 1.0) {
    throw std::invalid_argument("stitch fraction must lie in (0, 1]");
  }
  if (!std::isfinite(global_partner_probability) || global_partner_probability < 0.0 ||
      global_partner_probability > 1.0) {
    throw std::invalid_argument("global partner probability must lie in [0, 1]");
  }
  const double duration = fraction * model_.free.beta;
  double tau0 = 0.0;
  if (requested_tau0.has_value()) {
    tau0 = *requested_tau0;
  } else if (duration != model_.free.beta) {
    tau0 = random_.uniform_unit() * (model_.free.beta - duration);
  }
  const double tau1 = std::min(model_.free.beta, tau0 + duration);
  if (!std::isfinite(tau0) || tau0 < 0.0 || tau1 > model_.free.beta || tau1 <= tau0) {
    throw std::invalid_argument("stitch window lies outside [0, beta]");
  }

  const std::vector<Site> positions = state_.positions_at(tau0, model_.free);
  const PartnerBuckets buckets = build_partner_buckets(positions);
  const std::size_t update_count = updates.value_or(model_.free.particle_count);
  for (std::size_t update = 0; update < update_count; ++update) {
    const auto selected = static_cast<ParticleId>(
        random_.uniform_index(static_cast<std::uint64_t>(model_.free.particle_count)));
    const ParticleId partner =
        select_stitch_partner(selected, positions, buckets, model_.free, locality_radius,
                              global_partner_probability, random_);
    static_cast<void>(try_stitch_pair(selected, partner, tau0, tau1));
  }
}

bool InteractingSampler::time_shift_update(const std::optional<double> shift) {
  MoveStatistics &move_statistics = statistics_[move_index(MoveKind::TimeShiftMove)];
  increment_counter(move_statistics.attempts);
  const double selected_shift =
      shift.has_value() ? *shift : random_.uniform_unit() * model_.free.beta;
  ContinuousConfiguration rotated =
      rotate_configuration_time_origin(state_, model_.free, selected_shift);
  auto rebuilt = std::make_unique<detail::OccupancyIndex>(model_.free);
  rebuilt->rebuild(rotated.worldlines);
  ensure_counter_capacity(move_statistics.accepts);
  state_ = std::move(rotated);
  occupancy_index_ = std::move(rebuilt);
  increment_counter(move_statistics.accepts);
  return true;
}

void InteractingSampler::random_seam_stitch_sweep(const std::optional<std::size_t> updates,
                                                  const double fraction,
                                                  const std::size_t locality_radius,
                                                  const double global_partner_probability) {
  static_cast<void>(time_shift_update());
  stitch_sweep(updates, fraction, 0.0, locality_radius, global_partner_probability);
  static_cast<void>(time_shift_update());
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
  auto rebuilt = std::make_unique<detail::OccupancyIndex>(model_.free);
  rebuilt->rebuild(proposal.worldlines);
  ensure_counter_capacity(move_statistics.accepts);
  std::swap(state_, proposal);
  occupancy_index_ = std::move(rebuilt);
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
  if (options.stitch_updates != 0) {
    stitch_sweep(options.stitch_updates, options.stitch_fraction, std::nullopt,
                 options.stitch_locality_radius, options.stitch_global_partner_probability);
  }
  for (std::size_t update = 0; update < options.time_shift_updates; ++update) {
    static_cast<void>(time_shift_update());
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
