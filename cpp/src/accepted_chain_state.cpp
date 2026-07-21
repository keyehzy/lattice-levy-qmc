#include "accepted_chain_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace qmc::detail {
namespace {

bool nearly_equal_time(const double left, const double right) {
  if (left == right) {
    return true;
  }
  const double scale = std::max(std::abs(left), std::abs(right));
  return std::abs(left - right) <= 16.0 * std::numeric_limits<double>::epsilon() * scale;
}

void validate_overlap(const double overlap) {
  if (!std::isfinite(overlap)) {
    throw std::overflow_error("accepted pair-overlap integral is not finite");
  }
  if (overlap < 0.0) {
    throw std::logic_error("accepted pair-overlap integral became negative");
  }
}

} // namespace

AcceptedChainState::ReplacementTransaction::ReplacementTransaction(
    AcceptedChainState &owner, std::vector<PathReplacement> replacements,
    std::optional<Permutation> topology, const double proposed_overlap,
    OccupancyIndex::ReplacementTransaction occupancy_transaction) noexcept
    : owner_(&owner), replacements_(std::move(replacements)), topology_(std::move(topology)),
      proposed_overlap_(proposed_overlap),
      occupancy_transaction_(std::move(occupancy_transaction)) {}

AcceptedChainState::ReplacementTransaction::ReplacementTransaction(
    ReplacementTransaction &&other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), replacements_(std::move(other.replacements_)),
      topology_(std::move(other.topology_)), proposed_overlap_(other.proposed_overlap_),
      occupancy_transaction_(std::move(other.occupancy_transaction_)) {}

void AcceptedChainState::ReplacementTransaction::commit() noexcept {
  if (owner_ == nullptr) {
    return;
  }
  owner_->publish(*this);
  owner_ = nullptr;
}

AcceptedChainState::AcceptedChainState(ContinuousConfiguration configuration)
    : configuration_(std::move(configuration)),
      layout_(configuration_.model().linear_size(), configuration_.model().dimension()),
      occupancy_(configuration_.model()) {
  occupancy_.rebuild(configuration_.worldlines_);
  pair_overlap_ = occupancy_.pair_overlap();
  validate_overlap(pair_overlap_);
}

const ContinuousPath &
AcceptedChainState::path_after_replacement(const ParticleId label,
                                           const std::vector<PathReplacement> &replacements) const {
  const auto replacement =
      std::ranges::lower_bound(replacements, label, {}, &PathReplacement::label);
  if (replacement != replacements.end() && replacement->label == label) {
    return replacement->path;
  }
  return configuration_.worldlines_[label];
}

void AcceptedChainState::validate_replacement_inputs(
    const std::vector<PathReplacement> &replacements) const {
  for (std::size_t index = 0; index < replacements.size(); ++index) {
    const PathReplacement &replacement = replacements[index];
    if (static_cast<std::size_t>(replacement.label) >= configuration_.worldlines_.size() ||
        (index != 0 && replacements[index - 1].label == replacement.label)) {
      throw std::logic_error("path replacements contain an invalid or duplicate label");
    }
    if (replacement.path.dimension() != configuration_.model_.dimension()) {
      throw std::invalid_argument("replacement path dimension does not match the model");
    }
    if (!nearly_equal_time(replacement.path.duration(), configuration_.model_.beta())) {
      throw std::invalid_argument("replacement path duration does not match beta");
    }
  }
}

bool AcceptedChainState::replacement_endpoints_unchanged(
    const std::vector<PathReplacement> &replacements) const {
  return std::ranges::all_of(replacements, [this](const PathReplacement &replacement) {
    const ContinuousPath &accepted = configuration_.worldlines_[replacement.label];
    return layout_.encode_covering(replacement.path.start()) ==
               layout_.encode_covering(accepted.start()) &&
           layout_.encode_covering(replacement.path.end()) ==
               layout_.encode_covering(accepted.end());
  });
}

void AcceptedChainState::validate_replacement_connectivity(
    const std::vector<PathReplacement> &replacements, const Permutation &topology,
    const bool topology_changed) const {
  if (topology.size() != configuration_.worldlines_.size()) {
    throw std::logic_error("replacement topology size does not match the accepted state");
  }
  for (std::size_t particle = 0; particle < configuration_.worldlines_.size(); ++particle) {
    const auto label = static_cast<ParticleId>(particle);
    const ParticleId successor_label = topology.successor(label);
    if (!topology_changed &&
        !std::ranges::binary_search(replacements, label, {}, &PathReplacement::label) &&
        !std::ranges::binary_search(replacements, successor_label, {}, &PathReplacement::label)) {
      continue;
    }
    const ContinuousPath &path = path_after_replacement(label, replacements);
    const ContinuousPath &successor = path_after_replacement(successor_label, replacements);
    if (layout_.encode_covering(path.end()) != layout_.encode_covering(successor.start())) {
      throw std::logic_error("replacement paths do not join their permutation successors");
    }
  }
}

void AcceptedChainState::validate_replacement_state(
    const std::vector<PathReplacement> &replacements,
    const std::optional<Permutation> &topology) const {
  validate_replacement_inputs(replacements);
  if (!topology.has_value() && replacement_endpoints_unchanged(replacements)) {
    return;
  }
  validate_replacement_connectivity(replacements,
                                    topology.has_value() ? *topology : configuration_.topology(),
                                    topology.has_value());
}

AcceptedChainState::ReplacementTransaction
AcceptedChainState::begin_replacement(std::vector<PathReplacement> replacements,
                                      std::optional<std::vector<ParticleId>> successors) {
  std::ranges::sort(replacements, {}, &PathReplacement::label);
  std::optional<Permutation> topology;
  if (successors.has_value()) {
    topology.emplace(std::move(*successors));
  }
  validate_replacement_state(replacements, topology);

  std::vector<PathReplacementView> views;
  views.reserve(replacements.size());
  for (const PathReplacement &replacement : replacements) {
    views.push_back(PathReplacementView{
        .label = replacement.label,
        .old_path = configuration_.worldlines_[replacement.label],
        .new_path = replacement.path,
    });
  }
  auto occupancy_transaction = occupancy_.begin_replacement(views, pair_overlap_);
  double proposed_overlap = occupancy_transaction.proposed_overlap();
  if (proposed_overlap < 0.0) {
    proposed_overlap = occupancy_transaction.exact_proposed_overlap();
  }
  validate_overlap(proposed_overlap);
  return {*this, std::move(replacements), std::move(topology), proposed_overlap,
          std::move(occupancy_transaction)};
}

void AcceptedChainState::publish(ReplacementTransaction &transaction) noexcept {
  static_assert(std::is_nothrow_swappable_v<ContinuousPath>);
  static_assert(std::is_nothrow_move_assignable_v<Permutation>);
  for (PathReplacement &replacement : transaction.replacements_) {
    std::swap(configuration_.worldlines_[replacement.label], replacement.path);
  }
  if (transaction.topology_.has_value()) {
    configuration_.replace_topology(std::move(*transaction.topology_));
  }
  transaction.occupancy_transaction_.commit();
  pair_overlap_ = transaction.proposed_overlap();
}

bool AcceptedChainState::occupancy_matches_configuration() const {
  return occupancy_.represents(configuration_.worldlines_);
}

double AcceptedChainState::occupancy_pair_overlap() { return occupancy_.pair_overlap(); }

} // namespace qmc::detail
