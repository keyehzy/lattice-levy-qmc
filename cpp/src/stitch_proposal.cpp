#include "stitch_proposal.hpp"

#include "path_cursor.hpp"
#include "stitch_matching.hpp"
#include "stitch_seam_context.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace qmc::detail {
namespace {

ContinuousPath splice_path_interval(const PathSlice &prefix_slice, const PathSlice &suffix_slice,
                                    const ContinuousPath &bridge, const TorusLayout &layout) {
  if (bridge.start() != prefix_slice.start || bridge.end().size() != suffix_slice.end.size()) {
    throw std::invalid_argument("stitch bridge does not start at the prefix cut");
  }
  if (layout.encode_covering(bridge.end()) != layout.encode_covering(suffix_slice.end)) {
    throw std::invalid_argument("stitch bridge does not end at the suffix torus site");
  }
  return splice_path_slices(prefix_slice, suffix_slice, bridge);
}

} // namespace

StitchProposal sample_stitch_proposal(const ContinuousConfiguration &configuration,
                                      const std::span<const ParticleId> strands,
                                      StitchSeamContext &seam, Random &random) {
  const std::size_t strand_count = strands.size();
  if (strand_count < 2 || strand_count > kMaxStitchStrands) {
    throw std::invalid_argument("a stitch must contain between 2 and 8 strands");
  }

  std::vector<PathSlice> path_slices;
  std::vector<Site> left;
  std::vector<Site> right;
  path_slices.reserve(strand_count);
  left.reserve(strand_count);
  right.reserve(strand_count);
  for (std::size_t index = 0; index < strand_count; ++index) {
    const ParticleId label = strands[index];
    const std::span<const ParticleId> previous = strands.first(index);
    if (static_cast<std::size_t>(label) >= configuration.worldlines().size() ||
        std::ranges::find(previous, label) != previous.end()) {
      throw std::invalid_argument("stitch strands must be distinct valid labels");
    }
    const ContinuousPath &path = configuration.path(label);
    PathCursor cursor(path);
    const PathCut left_cut = cursor.cut(seam.tau0());
    const PathCut right_cut = cursor.cut(seam.tau1());
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
  const StitchMatching matching = PreparedPermanent(log_weights, strand_count).sample(random);

  StitchProposal proposal;
  proposal.replacements.reserve(strand_count);
  for (std::size_t row = 0; row < strand_count; ++row) {
    const std::size_t column = matching[row];
    const TorusBridgeDistribution &distribution =
        seam.bridge_distribution(left[row], right[column]);
    const Site covering_end = distribution.sample_covering_endpoint(left[row], random);
    const ContinuousPath bridge =
        sample_continuous_bridge(left[row], covering_end, seam.duration(), seam.kernels(), random);
    proposal.replacements.emplace_back(
        strands[row],
        splice_path_interval(path_slices[row], path_slices[column], bridge, seam.layout()));
  }

  const std::span<const ParticleId> current_successors = configuration.topology().successors();
  proposal.successors.assign(current_successors.begin(), current_successors.end());
  std::vector<ParticleId> old_successors;
  old_successors.reserve(strand_count);
  for (const ParticleId label : strands) {
    old_successors.push_back(configuration.topology().successor(label));
  }
  for (std::size_t row = 0; row < strand_count; ++row) {
    proposal.successors[strands[row]] = old_successors[matching[row]];
    if (matching[row] != row) {
      ++proposal.successor_changes;
    }
  }
  return proposal;
}

} // namespace qmc::detail
