#include "qmc/interacting_sampler.hpp"

#include "accepted_chain_state.hpp"
#include "continuous_detail.hpp"
#include "interaction_detail.hpp"
#include "path_cursor.hpp"
#include "stitch_matching.hpp"
#include "stitch_seam_context.hpp"

#include <algorithm>
#include <cassert>
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

void add_counter(std::uint64_t &counter, const std::size_t amount) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - counter) {
    throw std::overflow_error("move statistics counter overflowed");
  }
  counter += static_cast<std::uint64_t>(amount);
}

void ensure_counter_capacity(const std::uint64_t counter, const std::size_t amount) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - counter) {
    throw std::overflow_error("move statistics counter overflowed");
  }
}

double checked_action(const double interaction, const double overlap) {
  const double action = interaction * overlap;
  if (!std::isfinite(action)) {
    throw std::overflow_error("interaction action overflowed");
  }
  return action;
}

void validate_unit_fraction(const double fraction, const char *const message) {
  if (!std::isfinite(fraction) || fraction <= 0.0 || fraction > 1.0) {
    throw std::invalid_argument(message);
  }
}

void validate_stitch_controls(const double fraction, const double global_partner_probability) {
  validate_unit_fraction(fraction, "stitch fraction must lie in (0, 1]");
  if (!std::isfinite(global_partner_probability) || global_partner_probability < 0.0 ||
      global_partner_probability > 1.0) {
    throw std::invalid_argument("global partner probability must lie in [0, 1]");
  }
}

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
                                             const detail::StitchPartnerBuckets &buckets) {
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
                                                 const detail::StitchPartnerBuckets &buckets,
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
                                                const detail::StitchPartnerBuckets &buckets,
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

std::vector<ParticleId> unselected_labels(const std::vector<ParticleId> &pool,
                                          const std::vector<bool> &selected) {
  std::vector<ParticleId> result;
  result.reserve(pool.size());
  for (const ParticleId label : pool) {
    if (!selected[label]) {
      result.push_back(label);
    }
  }
  return result;
}

ParticleId draw_uniform_unselected(const std::vector<bool> &selected,
                                   const std::size_t selected_count, Random &random) {
  auto draw = static_cast<std::size_t>(
      random.uniform_index(static_cast<std::uint64_t>(selected.size() - selected_count)));
  for (std::size_t label = 0; label < selected.size(); ++label) {
    if (selected[label]) {
      continue;
    }
    if (draw == 0) {
      return static_cast<ParticleId>(label);
    }
    --draw;
  }
  throw std::logic_error("failed to draw an unselected stitch strand");
}

std::vector<ParticleId>
select_stitch_strands(const ParticleId anchor, const std::size_t strand_count,
                      const std::span<const SiteId> positions,
                      const detail::StitchPartnerBuckets &buckets, const TorusLayout &layout,
                      const Model &model, const std::size_t locality_radius,
                      const double global_partner_probability, Random &random) {
  assert(std::isfinite(global_partner_probability) && global_partner_probability >= 0.0 &&
         global_partner_probability <= 1.0);
  assert(strand_count >= 2 && strand_count <= detail::kMaxStitchStrands &&
         strand_count <= model.particle_count());
  assert(static_cast<std::size_t>(anchor) < model.particle_count());

  const std::vector<ParticleId> local_pool =
      local_stitch_candidates(anchor, positions, buckets, layout, locality_radius);
  std::vector<bool> selected(model.particle_count(), false);
  selected[anchor] = true;
  std::vector<ParticleId> strands{anchor};
  strands.reserve(strand_count);

  while (strands.size() < strand_count) {
    const bool use_global = random.uniform_unit() < global_partner_probability;
    const std::vector<ParticleId> local_candidates =
        use_global ? std::vector<ParticleId>{} : unselected_labels(local_pool, selected);

    ParticleId next = 0;
    if (!local_candidates.empty()) {
      next = local_candidates[random.uniform_index(
          static_cast<std::uint64_t>(local_candidates.size()))];
    } else {
      next = draw_uniform_unselected(selected, strands.size(), random);
    }
    selected[next] = true;
    strands.push_back(next);
  }
  return strands;
}

std::pair<double, double>
sample_stitch_window(const Model &model, Random &random,
                     const std::optional<std::pair<double, double>> interval,
                     const double fraction) {
  if (interval.has_value()) {
    return *interval;
  }
  const double duration = fraction * model.beta();
  const double tau0 =
      duration == model.beta() ? 0.0 : random.uniform_unit() * (model.beta() - duration);
  const double tau1 = std::min(model.beta(), tau0 + duration);
  return {tau0, tau1};
}

detail::StitchMatching sample_stitch_matching(const std::span<const double> log_weights,
                                              const std::size_t strand_count, Random &random) {
  return detail::PreparedPermanent(log_weights, strand_count).sample(random);
}

ContinuousPath splice_path_interval(const detail::PathSlice &prefix_slice,
                                    const detail::PathSlice &suffix_slice,
                                    const ContinuousPath &bridge, const Coord linear_size) {
  if (bridge.start() != prefix_slice.start || bridge.end().size() != suffix_slice.end.size()) {
    throw std::invalid_argument("stitch bridge does not start at the prefix cut");
  }
  const TorusLayout layout(linear_size, suffix_slice.end.size());
  if (layout.encode_covering(bridge.end()) != layout.encode_covering(suffix_slice.end)) {
    throw std::invalid_argument("stitch bridge does not end at the suffix torus site");
  }
  return detail::splice_path_slices(prefix_slice, suffix_slice, bridge);
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

std::optional<double> MoveStatistics::successor_changes_per_attempt() const noexcept {
  if (attempts == 0) {
    return std::nullopt;
  }
  return static_cast<double>(successor_changes) / static_cast<double>(attempts);
}

void InteractingSampler::validate_segment_update_options(
    const SegmentUpdateOptions &options) const {
  validate_unit_fraction(options.fraction, "segment fraction must lie in (0, 1]");
  if (options.particle.has_value() &&
      static_cast<std::size_t>(*options.particle) >= model_.free.particle_count()) {
    throw std::invalid_argument("particle label is out of range");
  }
  if (options.interval.has_value()) {
    const auto [tau0, tau1] = *options.interval;
    if (!std::isfinite(tau0) || !std::isfinite(tau1) || tau0 < 0.0 || tau1 > model_.free.beta() ||
        tau1 <= tau0) {
      throw std::invalid_argument("require 0 <= segment tau0 < tau1 <= beta");
    }
  }
}

void InteractingSampler::validate_stitch_update_options(const StitchUpdateOptions &options) const {
  validate_stitch_controls(options.fraction, options.global_partner_probability);
  if (options.strand_count < 2 || options.strand_count > detail::kMaxStitchStrands) {
    throw std::invalid_argument("stitch strand count must lie in [2, 8]");
  }
  if (options.interval.has_value()) {
    const auto [tau0, tau1] = *options.interval;
    if (!std::isfinite(tau0) || !std::isfinite(tau1) || tau0 < 0.0 || tau1 > model_.free.beta() ||
        tau1 <= tau0) {
      throw std::invalid_argument("require 0 <= stitch tau0 < tau1 <= beta");
    }
  }
  if (!options.strands.empty() && options.anchor.has_value()) {
    throw std::invalid_argument("explicit stitch strands cannot be combined with an anchor");
  }
  if (!options.strands.empty() && options.strands.size() != options.strand_count) {
    throw std::invalid_argument("explicit stitch strands must contain strand_count labels");
  }

  const std::size_t particle_count = model_.free.particle_count();
  if (options.anchor.has_value() && static_cast<std::size_t>(*options.anchor) >= particle_count) {
    throw std::invalid_argument("stitch anchor is out of range");
  }
  for (std::size_t index = 0; index < options.strands.size(); ++index) {
    const ParticleId label = options.strands[index];
    if (static_cast<std::size_t>(label) >= particle_count) {
      throw std::invalid_argument("stitch strands must be distinct valid labels");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (options.strands[previous] == label) {
        throw std::invalid_argument("stitch strands must be distinct valid labels");
      }
    }
  }

  // A valid pair request is a documented no-op when fewer than two particles
  // exist. Larger collective requests still report their impossible size.
  if (particle_count < 2) {
    if (options.strand_count != 2) {
      throw std::invalid_argument("stitch strand count cannot exceed the particle count");
    }
    return;
  }
  if (options.strand_count > particle_count) {
    throw std::invalid_argument("stitch strand count cannot exceed the particle count");
  }
}

InteractingSampler::PreparedStitchMixture
InteractingSampler::prepare_stitch_mixture(const StitchMixture &mixture) const {
  if (mixture.strand_counts.empty()) {
    throw std::invalid_argument("stitch strand counts must not be empty");
  }
  if (!mixture.strand_weights.empty() &&
      mixture.strand_weights.size() != mixture.strand_counts.size()) {
    throw std::invalid_argument("stitch strand weights must match strand counts");
  }

  std::array<bool, detail::kMaxStitchStrands + 1> seen{};
  PreparedStitchMixture prepared;
  for (std::size_t index = 0; index < mixture.strand_counts.size(); ++index) {
    const std::size_t count = mixture.strand_counts[index];
    if (count < 2 || count > detail::kMaxStitchStrands) {
      throw std::invalid_argument("every stitch strand count must lie in [2, 8]");
    }
    if (seen[count]) {
      throw std::invalid_argument("stitch strand counts must not contain duplicates");
    }
    seen[count] = true;
    const double weight = mixture.strand_weights.empty() ? 1.0 : mixture.strand_weights[index];
    if (!std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument("stitch strand weights must be finite and nonnegative");
    }
    if (count <= model_.free.particle_count()) {
      prepared.counts.push_back(count);
      prepared.weights.push_back(weight);
    }
  }
  if (prepared.counts.empty()) {
    return prepared;
  }
  double total = 0.0;
  for (const double weight : prepared.weights) {
    total += weight;
  }
  if (!std::isfinite(total) || total <= 0.0) {
    throw std::invalid_argument("valid stitch strand counts must have positive total weight");
  }
  return prepared;
}

InteractingSampler::PreparedStitchSweep
InteractingSampler::prepare_stitch_sweep(const StitchSweepOptions &options) const {
  validate_stitch_controls(options.fraction, options.global_partner_probability);

  const double duration = options.fraction * model_.free.beta();
  if (options.tau0.has_value()) {
    const double tau0 = *options.tau0;
    const double tau1 = std::min(model_.free.beta(), tau0 + duration);
    if (!std::isfinite(tau0) || tau0 < 0.0 || tau1 > model_.free.beta() || tau1 <= tau0) {
      throw std::invalid_argument("stitch window lies outside [0, beta]");
    }
  }

  return PreparedStitchSweep{
      .updates = options.updates.value_or(model_.free.particle_count()),
      .duration = duration,
      .tau0 = options.tau0,
      .locality_radius = options.locality_radius,
      .global_partner_probability = options.global_partner_probability,
      .mixture = prepare_stitch_mixture(options.mixture),
  };
}

InteractingSampler::PreparedSweep
InteractingSampler::prepare_sweep(const SweepOptions &options) const {
  validate_segment_update_options(SegmentUpdateOptions{
      .particle = std::nullopt,
      .interval = std::nullopt,
      .fraction = options.segment_fraction,
  });
  PreparedStitchSweep stitch = prepare_stitch_sweep(StitchSweepOptions{
      .updates = options.stitch_updates,
      .fraction = options.stitch_fraction,
      .tau0 = std::nullopt,
      .locality_radius = options.stitch_locality_radius,
      .global_partner_probability = options.stitch_global_partner_probability,
      .mixture = options.stitch_mixture,
  });
  return PreparedSweep{
      .segment_updates = options.segment_updates.value_or(model_.free.particle_count()),
      .segment_fraction = options.segment_fraction,
      .cycle_updates = options.cycle_updates,
      .global_updates = options.global_updates,
      .stitch = std::move(stitch),
      .time_shift_updates = options.time_shift_updates,
  };
}

InteractingSampler::InteractingSampler(InteractingModel model, NumericalOptions numerical,
                                       const std::uint64_t seed)
    : model_(model), layout_(model_.free.linear_size(), model_.free.dimension()), random_(seed),
      free_ensemble_(model_.free, numerical) {
  model_.validate();
  ContinuousConfiguration initial = sample_ideal_continuous_configuration(free_ensemble_, random_);
  accepted_state_ = std::make_unique<detail::AcceptedChainState>(std::move(initial));
  static_cast<void>(checked_action(model_.interaction, pair_overlap()));
}

InteractingSampler::~InteractingSampler() = default;
InteractingSampler::InteractingSampler(const InteractingSampler &other)
    : model_(other.model_), layout_(other.layout_), random_(other.random_),
      free_ensemble_(other.free_ensemble_),
      accepted_state_(std::make_unique<detail::AcceptedChainState>(*other.accepted_state_)),
      statistics_(other.statistics_) {}

InteractingSampler &InteractingSampler::operator=(const InteractingSampler &other) {
  if (this != &other) {
    InteractingSampler copy(other);
    *this = std::move(copy);
  }
  return *this;
}

InteractingSampler::InteractingSampler(InteractingSampler &&) noexcept = default;
InteractingSampler &InteractingSampler::operator=(InteractingSampler &&) noexcept = default;

const ContinuousConfiguration &InteractingSampler::state() const noexcept {
  return accepted_state_->configuration();
}

double InteractingSampler::pair_overlap() const noexcept { return accepted_state_->pair_overlap(); }

double InteractingSampler::action() const noexcept { return model_.interaction * pair_overlap(); }

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
  MoveStatistics &move_statistics = statistics_[move_index(move)];
  increment_counter(move_statistics.attempts);
  return try_proposal(
      LocalProposal{
          .replacements = std::move(replacements),
          .successors = std::nullopt,
          .successor_changes = 0,
      },
      move_statistics);
}

bool InteractingSampler::try_proposal(LocalProposal proposal, MoveStatistics &move_statistics) {
  if (!proposal.successors.has_value() && proposal.successor_changes != 0) {
    throw std::logic_error("path-only proposal reports topology changes");
  }
  std::vector<detail::AcceptedChainState::PathReplacement> replacements;
  replacements.reserve(proposal.replacements.size());
  for (LabeledPath &replacement : proposal.replacements) {
    replacements.push_back(detail::AcceptedChainState::PathReplacement{
        .label = replacement.first,
        .path = std::move(replacement.second),
    });
  }
  auto state_transaction =
      accepted_state_->begin_replacement(std::move(replacements), std::move(proposal.successors));

  const double new_overlap = state_transaction.proposed_overlap();
  const double new_action = checked_action(model_.interaction, new_overlap);
  const bool accepted = metropolis_accept(new_action - action());
  if (!accepted) {
    return false;
  }

  ensure_counter_capacity(move_statistics.accepts);
  if (proposal.successor_changes != 0) {
    ensure_counter_capacity(move_statistics.topology_changes);
    ensure_counter_capacity(move_statistics.successor_changes, proposal.successor_changes);
  }

  state_transaction.commit();
  increment_counter(move_statistics.accepts);
  if (proposal.successor_changes != 0) {
    increment_counter(move_statistics.topology_changes);
    add_counter(move_statistics.successor_changes, proposal.successor_changes);
  }
  return true;
}

bool InteractingSampler::segment_update(const SegmentUpdateOptions &options) {
  validate_segment_update_options(options);
  return execute_segment_update(options);
}

bool InteractingSampler::execute_segment_update(const SegmentUpdateOptions &options) {
  if (model_.free.particle_count() == 0) {
    return true;
  }
  const ParticleId label = options.particle.has_value()
                               ? *options.particle
                               : static_cast<ParticleId>(random_.uniform_index(
                                     static_cast<std::uint64_t>(model_.free.particle_count())));

  double tau0 = 0.0;
  double tau1 = 0.0;
  if (options.interval.has_value()) {
    tau0 = options.interval->first;
    tau1 = options.interval->second;
  } else {
    const double duration = options.fraction * model_.free.beta();
    if (duration == model_.free.beta()) {
      tau0 = 0.0;
    } else {
      tau0 = random_.uniform_unit() * (model_.free.beta() - duration);
    }
    tau1 = std::min(model_.free.beta(), tau0 + duration);
  }

  ContinuousPath proposal = resample_path_interval(state().path(label), tau0, tau1,
                                                   free_ensemble_.free_path_kernels(), random_);
  std::vector<LabeledPath> replacements;
  replacements.emplace_back(label, std::move(proposal));
  return try_path_replacements(std::move(replacements), MoveKind::SegmentMove);
}

bool InteractingSampler::whole_worldline_update(const std::optional<ParticleId> particle) {
  return segment_update(SegmentUpdateOptions{
      .particle = particle,
      .interval = std::pair<double, double>{0.0, model_.free.beta()},
  });
}

bool InteractingSampler::cycle_update(const std::optional<std::size_t> cycle_index) {
  const std::span<const Cycle> cycles = state().topology().cycles();
  if (cycles.empty()) {
    return true;
  }
  const std::size_t selected = cycle_index.has_value()
                                   ? *cycle_index
                                   : static_cast<std::size_t>(random_.uniform_index(
                                         static_cast<std::uint64_t>(cycles.size())));
  if (selected >= cycles.size()) {
    throw std::invalid_argument("cycle index is out of range");
  }
  const Cycle &cycle = cycles[selected];
  auto proposed_paths = detail::sample_paths_for_cycle(cycle, model_.free,
                                                       free_ensemble_.free_path_kernels(), random_);
  std::vector<LabeledPath> replacements;
  replacements.reserve(cycle.size());
  for (std::size_t index = 0; index < cycle.size(); ++index) {
    replacements.emplace_back(cycle[index], std::move(proposed_paths[index]));
  }
  return try_path_replacements(std::move(replacements), MoveKind::CycleMove);
}

InteractingSampler::StitchProposal
InteractingSampler::sample_stitch_proposal(const std::span<const ParticleId> strands,
                                           detail::StitchSeamContext &seam) {
  const std::size_t strand_count = strands.size();
  if (strand_count < 2 || strand_count > detail::kMaxStitchStrands) {
    throw std::invalid_argument("a stitch must contain between 2 and 8 strands");
  }
  std::vector<bool> selected(state().worldlines().size(), false);
  std::vector<detail::PathSlice> path_slices;
  std::vector<Site> left;
  std::vector<Site> right;
  path_slices.reserve(strand_count);
  left.reserve(strand_count);
  right.reserve(strand_count);
  for (const ParticleId label : strands) {
    if (static_cast<std::size_t>(label) >= state().worldlines().size() || selected[label]) {
      throw std::invalid_argument("stitch strands must be distinct valid labels");
    }
    selected[label] = true;
    const ContinuousPath &path = state().path(label);
    detail::PathCursor cursor(path);
    const detail::PathCut left_cut = cursor.cut(seam.tau0());
    const detail::PathCut right_cut = cursor.cut(seam.tau1());
    path_slices.push_back(cursor.slice(left_cut, right_cut));
    left.push_back(path_slices.back().start);
    right.push_back(path_slices.back().end);
  }

  std::vector<double> log_weights(strand_count * strand_count);
  for (std::size_t row = 0; row < strand_count; ++row) {
    for (std::size_t column = 0; column < strand_count; ++column) {
      log_weights[(row * strand_count) + column] =
          seam.bridge_distribution(left[row], right[column]).log_normalization();
    }
  }
  const detail::StitchMatching matching =
      sample_stitch_matching(log_weights, strand_count, random_);

  StitchProposal proposal;
  proposal.replacements.reserve(strand_count);
  for (std::size_t row = 0; row < strand_count; ++row) {
    const std::size_t column = matching[row];
    const detail::TorusBridgeDistribution &distribution =
        seam.bridge_distribution(left[row], right[column]);
    const Site covering_end = distribution.sample_covering_endpoint(left[row], random_);
    const ContinuousPath bridge = sample_continuous_bridge(
        left[row], covering_end, seam.duration(), free_ensemble_.free_path_kernels(), random_);
    proposal.replacements.emplace_back(strands[row],
                                       splice_path_interval(path_slices[row], path_slices[column],
                                                            bridge, model_.free.linear_size()));
  }

  const std::span<const ParticleId> current_successors = state().topology().successors();
  proposal.successors.assign(current_successors.begin(), current_successors.end());
  std::vector<ParticleId> old_successors;
  old_successors.reserve(strand_count);
  for (const ParticleId label : strands) {
    old_successors.push_back(state().topology().successor(label));
  }
  for (std::size_t row = 0; row < strand_count; ++row) {
    proposal.successors[strands[row]] = old_successors[matching[row]];
    if (matching[row] != row) {
      ++proposal.successor_changes;
    }
  }
  return proposal;
}

bool InteractingSampler::try_stitch_strands(const std::span<const ParticleId> strands,
                                            detail::StitchSeamContext &seam) {
  MoveStatistics &move_statistics = statistics_[move_index(MoveKind::StitchMove)];
  increment_counter(move_statistics.attempts);
  StitchProposal proposal = sample_stitch_proposal(strands, seam);
  return try_proposal(
      LocalProposal{
          .replacements = std::move(proposal.replacements),
          .successors = std::move(proposal.successors),
          .successor_changes = proposal.successor_changes,
      },
      move_statistics);
}

bool InteractingSampler::stitch_update(const StitchUpdateOptions &options) {
  validate_stitch_update_options(options);
  return execute_stitch_update(options);
}

bool InteractingSampler::execute_stitch_update(const StitchUpdateOptions &options) {
  if (model_.free.particle_count() < 2) {
    return true;
  }
  const auto [tau0, tau1] =
      sample_stitch_window(model_.free, random_, options.interval, options.fraction);
  detail::StitchSeamContext seam(state(), tau0, tau1, layout_, free_ensemble_.free_path_kernels());

  std::vector<ParticleId> strands;
  if (options.strands.empty()) {
    const ParticleId anchor = options.anchor.has_value()
                                  ? *options.anchor
                                  : static_cast<ParticleId>(random_.uniform_index(
                                        static_cast<std::uint64_t>(model_.free.particle_count())));
    strands = select_stitch_strands(
        anchor, options.strand_count, seam.left_site_ids(), seam.partner_buckets(), layout_,
        model_.free, options.locality_radius, options.global_partner_probability, random_);
  } else {
    strands = options.strands;
  }
  return try_stitch_strands(strands, seam);
}

void InteractingSampler::stitch_sweep(const StitchSweepOptions &options) {
  execute_stitch_sweep(prepare_stitch_sweep(options));
}

void InteractingSampler::execute_stitch_sweep(const PreparedStitchSweep &options) {
  if (options.updates == 0 || options.mixture.counts.empty()) {
    return;
  }
  double tau0 = 0.0;
  if (options.tau0.has_value()) {
    tau0 = *options.tau0;
  } else if (options.duration != model_.free.beta()) {
    tau0 = random_.uniform_unit() * (model_.free.beta() - options.duration);
  }
  const double tau1 = std::min(model_.free.beta(), tau0 + options.duration);

  detail::StitchSeamContext seam(state(), tau0, tau1, layout_, free_ensemble_.free_path_kernels());
  for (std::size_t update = 0; update < options.updates; ++update) {
    const std::size_t mixture_index =
        options.mixture.counts.size() == 1 ? 0 : random_.discrete_index(options.mixture.weights);
    const std::size_t strand_count = options.mixture.counts[mixture_index];
    const auto anchor = static_cast<ParticleId>(
        random_.uniform_index(static_cast<std::uint64_t>(model_.free.particle_count())));
    const std::vector<ParticleId> strands = select_stitch_strands(
        anchor, strand_count, seam.left_site_ids(), seam.partner_buckets(), layout_, model_.free,
        options.locality_radius, options.global_partner_probability, random_);
    static_cast<void>(try_stitch_strands(strands, seam));
  }
}

bool InteractingSampler::time_shift_update(const std::optional<double> shift) {
  MoveStatistics &move_statistics = statistics_[move_index(MoveKind::TimeShiftMove)];
  increment_counter(move_statistics.attempts);
  const double selected_shift =
      shift.has_value() ? *shift : random_.uniform_unit() * model_.free.beta();
  ContinuousConfiguration rotated = rotate_configuration_time_origin(state(), selected_shift);
  auto proposed_state = std::make_unique<detail::AcceptedChainState>(std::move(rotated));
  static_cast<void>(checked_action(model_.interaction, proposed_state->pair_overlap()));
  ensure_counter_capacity(move_statistics.accepts);
  accepted_state_.swap(proposed_state);
  increment_counter(move_statistics.accepts);
  return true;
}

void InteractingSampler::random_seam_stitch_sweep(const RandomSeamStitchOptions &options) {
  const PreparedStitchSweep prepared = prepare_stitch_sweep(StitchSweepOptions{
      .updates = options.updates,
      .fraction = options.fraction,
      .tau0 = 0.0,
      .locality_radius = options.locality_radius,
      .global_partner_probability = options.global_partner_probability,
      .mixture = options.mixture,
  });
  static_cast<void>(time_shift_update());
  execute_stitch_sweep(prepared);
  static_cast<void>(time_shift_update());
}

bool InteractingSampler::global_update() {
  ContinuousConfiguration proposal = sample_ideal_continuous_configuration(free_ensemble_, random_);
  auto proposed_state = std::make_unique<detail::AcceptedChainState>(std::move(proposal));
  const double new_overlap = proposed_state->pair_overlap();
  const double new_action = checked_action(model_.interaction, new_overlap);
  MoveStatistics &move_statistics = statistics_[move_index(MoveKind::GlobalMove)];
  increment_counter(move_statistics.attempts);
  const bool accepted = metropolis_accept(new_action - action());
  if (!accepted) {
    return false;
  }
  ensure_counter_capacity(move_statistics.accepts);
  accepted_state_.swap(proposed_state);
  increment_counter(move_statistics.accepts);
  return true;
}

void InteractingSampler::sweep(const SweepOptions &options) {
  execute_sweep(prepare_sweep(options));
}

void InteractingSampler::execute_sweep(const PreparedSweep &options) {
  const SegmentUpdateOptions segment_options{
      .particle = std::nullopt,
      .interval = std::nullopt,
      .fraction = options.segment_fraction,
  };
  for (std::size_t update = 0; update < options.segment_updates; ++update) {
    static_cast<void>(execute_segment_update(segment_options));
  }
  for (std::size_t update = 0; update < options.cycle_updates; ++update) {
    static_cast<void>(cycle_update());
  }
  execute_stitch_sweep(options.stitch);
  for (std::size_t update = 0; update < options.time_shift_updates; ++update) {
    static_cast<void>(time_shift_update());
  }
  for (std::size_t update = 0; update < options.global_updates; ++update) {
    static_cast<void>(global_update());
  }
}

InteractingObservables InteractingSampler::observables() const {
  const double overlap = pair_overlap();
  const InteractionMeasurement interaction =
      detail::interaction_measurement_from_validated_overlap(state(), model_, overlap);
  return InteractingObservables{
      .action = interaction.action,
      .pair_overlap_time = interaction.pair_overlap_time,
      .double_occupancy_per_site = interaction.double_occupancy_per_site,
      .kinetic_energy = interaction.kinetic_energy,
      .interaction_energy = interaction.interaction_energy,
      .total_energy = interaction.total_energy,
      .event_count = interaction.event_count,
      .winding = state().total_winding(),
      .cycle_lengths = state().cycle_lengths(),
  };
}

std::vector<InteractingObservables> InteractingSampler::run(const std::size_t sample_count,
                                                            const RunOptions &options) {
  if (options.thin == 0) {
    throw std::invalid_argument("run thinning interval must be positive");
  }
  const PreparedSweep prepared = prepare_sweep(options.sweep);
  std::vector<InteractingObservables> output;
  if (sample_count > output.max_size()) {
    throw std::length_error("sample count exceeds vector capacity");
  }
  for (std::size_t sweep_index = 0; sweep_index < options.burn_in; ++sweep_index) {
    execute_sweep(prepared);
  }

  output.reserve(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    for (std::size_t thin_index = 0; thin_index < options.thin; ++thin_index) {
      execute_sweep(prepared);
    }
    output.push_back(observables());
  }
  return output;
}

} // namespace qmc
