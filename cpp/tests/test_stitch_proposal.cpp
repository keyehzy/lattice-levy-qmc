#include "qmc/continuous_configuration.hpp"
#include "qmc/permutation.hpp"
#include "qmc/torus_layout.hpp"
#include "stitch_proposal.hpp"
#include "stitch_seam_context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc {
namespace {

Model stitch_proposal_model() {
  return Model(ModelParameters{
      .particle_count = 6,
      .beta = 1.1,
      .linear_size = 5,
      .dimension = 2,
      .hopping = 0.8,
  });
}

TEST(StitchProposalTest, ProducesCompleteValidFixedSeamProposal) {
  const Model model = stitch_proposal_model();
  const CanonicalEnsemble ensemble(model);
  Random configuration_random(3'401);
  const ContinuousConfiguration configuration =
      sample_ideal_continuous_configuration(ensemble, configuration_random);
  const ContinuousConfiguration original = configuration;
  const TorusLayout layout(model.linear_size(), model.dimension());
  constexpr double tau0 = 0.2;
  constexpr double tau1 = 0.8;
  detail::StitchSeamContext seam(configuration, tau0, tau1, layout, ensemble.free_path_kernels());
  const std::array<ParticleId, 4> strands{0, 2, 4, 5};
  Random proposal_random(8'129);
  Random repeated_random(8'129);
  detail::StitchSeamContext repeated_seam(configuration, tau0, tau1, layout,
                                          ensemble.free_path_kernels());

  const detail::StitchProposal proposal =
      detail::sample_stitch_proposal(configuration, strands, seam, proposal_random);
  const detail::StitchProposal repeated =
      detail::sample_stitch_proposal(configuration, strands, repeated_seam, repeated_random);

  ASSERT_EQ(proposal.replacements.size(), strands.size());
  ASSERT_EQ(proposal.successors.size(), model.particle_count());
  EXPECT_EQ(proposal.replacements, repeated.replacements);
  EXPECT_EQ(proposal.successors, repeated.successors);
  EXPECT_EQ(proposal.successor_changes, repeated.successor_changes);
  EXPECT_DOUBLE_EQ(proposal_random.uniform_unit(), repeated_random.uniform_unit());
  std::vector<bool> selected(model.particle_count(), false);
  std::vector<ContinuousPath> proposed_paths(configuration.worldlines().begin(),
                                             configuration.worldlines().end());
  for (std::size_t index = 0; index < proposal.replacements.size(); ++index) {
    const auto &[label, path] = proposal.replacements[index];
    EXPECT_EQ(label, strands[index]);
    selected[static_cast<std::size_t>(label)] = true;
    proposed_paths[static_cast<std::size_t>(label)] = path;
  }

  std::size_t successor_changes = 0;
  for (std::size_t label = 0; label < model.particle_count(); ++label) {
    const ParticleId particle = static_cast<ParticleId>(label);
    if (!selected[label]) {
      EXPECT_EQ(proposal.successors[label], configuration.topology().successor(particle));
    }
    successor_changes +=
        proposal.successors[label] != configuration.topology().successor(particle) ? 1U : 0U;
  }
  EXPECT_EQ(proposal.successor_changes, successor_changes);

  const ContinuousConfiguration proposed(model, Permutation(proposal.successors),
                                         std::move(proposed_paths));
  EXPECT_TRUE(seam.matches_left_positions(proposed));
  EXPECT_EQ(configuration, original);
  EXPECT_GT(seam.cached_distribution_count(), 0U);
}

TEST(StitchProposalTest, RejectsInvalidStrandsBeforeDrawing) {
  const Model model = stitch_proposal_model();
  const CanonicalEnsemble ensemble(model);
  Random configuration_random(4'219);
  const ContinuousConfiguration configuration =
      sample_ideal_continuous_configuration(ensemble, configuration_random);
  const TorusLayout layout(model.linear_size(), model.dimension());
  detail::StitchSeamContext seam(configuration, 0.15, 0.75, layout, ensemble.free_path_kernels());

  const auto expect_invalid_without_draw = [&](const std::span<const ParticleId> strands) {
    Random actual(91'013);
    Random expected(91'013);
    EXPECT_THROW(
        static_cast<void>(detail::sample_stitch_proposal(configuration, strands, seam, actual)),
        std::invalid_argument);
    EXPECT_DOUBLE_EQ(actual.uniform_unit(), expected.uniform_unit());
  };

  const std::array<ParticleId, 1> too_few{0};
  const std::array<ParticleId, 9> too_many{0, 1, 2, 3, 4, 5, 0, 1, 2};
  const std::array<ParticleId, 2> duplicate{2, 2};
  const std::array<ParticleId, 2> out_of_range{0, 6};
  expect_invalid_without_draw(too_few);
  expect_invalid_without_draw(too_many);
  expect_invalid_without_draw(duplicate);
  expect_invalid_without_draw(out_of_range);
}

} // namespace
} // namespace qmc
