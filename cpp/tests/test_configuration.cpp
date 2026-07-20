#include "qmc/configuration.hpp"
#include "qmc/torus_layout.hpp"

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

static_assert(std::is_same_v<decltype(std::declval<const qmc::IdealBosonConfiguration &>().model()),
                             const qmc::Model &>);
static_assert(
    std::is_same_v<decltype(std::declval<const qmc::IdealBosonConfiguration &>().topology()),
                   const qmc::Permutation &>);
static_assert(std::is_same_v<
              decltype(std::declval<const qmc::IdealBosonConfiguration &>().covering_worldlines()),
              const qmc::DenseWorldlines &>);

TEST(DenseWorldlinesTest, StoresFlatRowMajorValuesWithBoundsChecking) {
  qmc::DenseWorldlines lines(2, 3, 2);
  lines.at(1, 2, 1) = 17;
  EXPECT_EQ(lines.at(1, 2, 1), 17);
  EXPECT_EQ(lines.values().size(), 12U);
  EXPECT_EQ(lines.particle_count(), 2U);
  EXPECT_EQ(lines.time_points(), 3U);
  EXPECT_EQ(lines.dimension(), 2U);
  EXPECT_EQ(lines.site(1, 2)[1], 17);
  EXPECT_THROW(static_cast<void>(lines.at(2, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(lines.at(0, 0, 2)), std::out_of_range);
}

TEST(DenseWorldlinesTest, RejectsMalformedConstructionInsteadOfExposingItsShape) {
  EXPECT_THROW(static_cast<void>(qmc::DenseWorldlines(2, 3, 2, std::vector<qmc::Coord>(11))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::DenseWorldlines(2, 3, 0)), std::invalid_argument);
}

TEST(IdealConfigurationTest, AssemblesConsistentAuthoritativeWorldLines) {
  const qmc::Model model{
      .particle_count = 7,
      .beta = 1.2,
      .linear_size = 6,
      .dimension = 2,
      .hopping = 0.9,
  };
  qmc::Random random(2026);
  const auto configuration = qmc::sample_ideal_boson_configuration(model, 7, random);
  EXPECT_NO_THROW(configuration.validate());
  EXPECT_EQ(configuration.model(), model);
  EXPECT_EQ(configuration.time_links_per_beta(), 7U);
  EXPECT_EQ(configuration.covering_worldlines().particle_count(), 7U);
  EXPECT_EQ(configuration.covering_worldlines().time_points(), 8U);
  EXPECT_EQ(configuration.covering_worldlines().dimension(), 2U);

  const qmc::TorusLayout layout(model.linear_size, model.dimension);
  std::vector<qmc::ParticleId> labels;
  const auto cycles = configuration.topology().cycles();
  for (std::size_t cycle_index = 0; cycle_index < cycles.size(); ++cycle_index) {
    const qmc::Cycle &cycle = cycles[cycle_index];
    labels.insert(labels.end(), cycle.begin(), cycle.end());
    const qmc::Site winding = configuration.cycle_winding(cycle_index);
    const auto root = cycle.front();
    const auto last = cycle.back();
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      EXPECT_EQ(configuration.covering_worldlines().at(last, 7, axis) -
                    configuration.covering_worldlines().at(root, 0, axis),
                model.linear_size * winding[axis]);
    }
  }
  for (std::size_t particle = 0; particle < model.particle_count; ++particle) {
    const auto label = static_cast<qmc::ParticleId>(particle);
    EXPECT_EQ(layout.encode_covering(configuration.covering_worldlines().site(label, 7)),
              layout.encode_covering(configuration.covering_worldlines().site(
                  configuration.topology().successor(label), 0)));
  }
  std::ranges::sort(labels);
  for (std::size_t index = 0; index < labels.size(); ++index) {
    EXPECT_EQ(labels[index], index);
  }
}

TEST(IdealConfigurationTest, SupportsEmptySystem) {
  const qmc::Model model{
      .particle_count = 0,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 2,
      .hopping = 1.0,
  };
  qmc::Random random(3);
  const auto configuration = qmc::sample_ideal_boson_configuration(model, 5, random);
  EXPECT_TRUE(configuration.topology().empty());
  EXPECT_TRUE(configuration.topology().cycles().empty());
  EXPECT_TRUE(configuration.covering_worldlines().values().empty());
  EXPECT_NO_THROW(configuration.validate());
}

TEST(IdealConfigurationTest, ZeroHoppingProducesConstantCoveringPaths) {
  const qmc::Model model{
      .particle_count = 8,
      .beta = 2.0,
      .linear_size = 5,
      .dimension = 3,
      .hopping = 0.0,
  };
  qmc::Random random(91);
  const auto configuration = qmc::sample_ideal_boson_configuration(model, 9, random);
  const auto &worldlines = configuration.covering_worldlines();
  const auto cycles = configuration.topology().cycles();
  for (std::size_t cycle_index = 0; cycle_index < cycles.size(); ++cycle_index) {
    const qmc::Site winding = configuration.cycle_winding(cycle_index);
    EXPECT_TRUE(std::ranges::all_of(winding, [](const qmc::Coord value) { return value == 0; }));
    const auto origin = worldlines.site(cycles[cycle_index].front(), 0);
    for (const qmc::ParticleId label : cycles[cycle_index]) {
      for (std::size_t time = 0; time <= configuration.time_links_per_beta(); ++time) {
        EXPECT_TRUE(std::ranges::equal(origin, worldlines.site(label, time)));
      }
    }
  }
}

TEST(IdealConfigurationTest, ConstructionRejectsMismatchedTopologyAndDisconnectedEndpoints) {
  const qmc::Model model{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 6,
      .dimension = 1,
      .hopping = 1.0,
  };
  EXPECT_THROW(static_cast<void>(qmc::IdealBosonConfiguration(model, 1, qmc::Permutation({0}),
                                                              qmc::DenseWorldlines(2, 2, 1))),
               std::invalid_argument);

  qmc::DenseWorldlines disconnected(2, 2, 1);
  disconnected.at(0, 1, 0) = model.linear_size;
  EXPECT_THROW(static_cast<void>(qmc::IdealBosonConfiguration(model, 1, qmc::Permutation({1, 0}),
                                                              std::move(disconnected))),
               std::invalid_argument);
}

TEST(IdealConfigurationTest, DerivesSignedWindingAndRejectsUnrepresentableDisplacement) {
  const qmc::Model model{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 1.0,
  };
  qmc::DenseWorldlines worldlines(2, 2, 1);
  worldlines.at(0, 0, 0) = 1;
  worldlines.at(0, 1, 0) = 3;
  worldlines.at(1, 0, 0) = 3;
  worldlines.at(1, 1, 0) = -3;
  const qmc::IdealBosonConfiguration configuration(model, 1, qmc::Permutation({1, 0}),
                                                   std::move(worldlines));
  EXPECT_EQ(configuration.cycle_winding(0), qmc::Site({-1}));

  const qmc::Model unit_torus{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 1,
      .dimension = 1,
      .hopping = 1.0,
  };
  qmc::DenseWorldlines unrepresentable(1, 2, 1);
  unrepresentable.at(0, 0, 0) = std::numeric_limits<qmc::Coord>::max();
  unrepresentable.at(0, 1, 0) = std::numeric_limits<qmc::Coord>::min();
  EXPECT_THROW(static_cast<void>(qmc::IdealBosonConfiguration(unit_torus, 1, qmc::Permutation({0}),
                                                              std::move(unrepresentable))),
               std::overflow_error);
}

TEST(IdealConfigurationTest, RejectsInvalidLayoutBeforeConsumingRandomness) {
  const qmc::Model model{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = std::numeric_limits<std::size_t>::digits,
      .hopping = 0.0,
  };
  const qmc::CanonicalEnsemble ensemble(model);
  qmc::Random random(93);
  qmc::Random control(93);
  EXPECT_THROW(static_cast<void>(qmc::sample_ideal_boson_configuration(ensemble, 1, random)),
               std::overflow_error);
  EXPECT_EQ(random.uniform_index(1'000), control.uniform_index(1'000));
}

TEST(IdealConfigurationTest, ReusableEnsembleMatchesOneOffWrapperForTheSameSeed) {
  const qmc::Model model{
      .particle_count = 6,
      .beta = 0.9,
      .linear_size = 8,
      .dimension = 2,
      .hopping = 1.1,
  };
  const qmc::CanonicalEnsemble ensemble(model);
  qmc::Random first_random(8128);
  qmc::Random second_random(8128);
  const auto first = qmc::sample_ideal_boson_configuration(model, 11, first_random);
  const auto second = qmc::sample_ideal_boson_configuration(ensemble, 11, second_random);

  EXPECT_EQ(first, second);
}

TEST(IdealConfigurationTest, RepeatedSamplingHandlesNegativeUnitWinding) {
  const qmc::Model model{
      .particle_count = 4,
      .beta = 0.8,
      .linear_size = 5,
      .dimension = 2,
      .hopping = 0.9,
  };
  qmc::Random random(2026);
  for (std::size_t sample = 0; sample < 20; ++sample) {
    EXPECT_NO_THROW(static_cast<void>(qmc::sample_ideal_boson_configuration(model, 8, random)));
  }
}

} // namespace
