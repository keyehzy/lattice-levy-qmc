#include "qmc/continuous_configuration.hpp"

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>

namespace qmc {
namespace {

TEST(ContinuousConfigurationTest, SamplesConsistentCanonicalState) {
  const Model model{
      .particle_count = 8,
      .beta = 1.3,
      .linear_size = 7,
      .dimension = 2,
      .hopping = 0.75,
  };
  Random random(11);
  const auto state = sample_ideal_continuous_configuration(model, random);
  EXPECT_NO_THROW(state.validate(model));
  EXPECT_EQ(state.permutation.size(), 8U);
  EXPECT_EQ(state.worldlines.size(), 8U);
  const auto lengths = state.cycle_lengths();
  EXPECT_EQ(std::accumulate(lengths.begin(), lengths.end(), std::size_t{0}), 8U);
  EXPECT_EQ(state.total_winding(model).size(), 2U);
  for (const ContinuousPath &path : state.worldlines) {
    EXPECT_DOUBLE_EQ(path.duration, model.beta);
  }
  for (const Site &position : state.positions_at(0.7, model)) {
    ASSERT_EQ(position.size(), 2U);
    EXPECT_TRUE(std::ranges::all_of(
        position, [](const Coord coordinate) { return coordinate >= 0 && coordinate < 7; }));
  }
}

TEST(ContinuousConfigurationTest, SupportsEmptyCanonicalState) {
  const Model model{
      .particle_count = 0,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 2,
      .hopping = 1.0,
  };
  Random random(3);
  const auto state = sample_ideal_continuous_configuration(model, random);
  EXPECT_NO_THROW(state.validate(model));
  EXPECT_TRUE(state.cycles.empty());
  EXPECT_TRUE(state.worldlines.empty());
  EXPECT_EQ(state.event_count(), 0U);
  EXPECT_EQ(state.total_winding(model), Site({0, 0}));
}

TEST(ContinuousConfigurationTest, ValidationDetectsTopologyAndEndpointCorruption) {
  const Model model{
      .particle_count = 4,
      .beta = 1.0,
      .linear_size = 5,
      .dimension = 1,
      .hopping = 0.8,
  };
  Random random(8);
  auto state = sample_ideal_continuous_configuration(model, random);
  state.permutation[0] = state.permutation[1];
  EXPECT_THROW(state.validate(model), std::logic_error);

  state = sample_ideal_continuous_configuration(model, random);
  ++state.worldlines[0].end[0];
  EXPECT_THROW(state.validate(model), std::invalid_argument);
}

TEST(ContinuousConfigurationTest, RejectsZeroBetaContinuousSampling) {
  const Model model{
      .particle_count = 1,
      .beta = 0.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  };
  Random random(9);
  EXPECT_THROW(static_cast<void>(sample_ideal_continuous_configuration(model, random)),
               std::invalid_argument);
}

} // namespace
} // namespace qmc
