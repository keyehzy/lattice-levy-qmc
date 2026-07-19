#include "qmc/continuous_configuration.hpp"
#include "qmc/interaction.hpp"

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>

namespace qmc {
namespace {

bool paths_are_equal(const ContinuousPath &left, const ContinuousPath &right) {
  if (left.duration != right.duration || left.start != right.start || left.end != right.end ||
      left.events.size() != right.events.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.events.size(); ++index) {
    if (left.events[index].time != right.events[index].time ||
        left.events[index].axis != right.events[index].axis ||
        left.events[index].direction != right.events[index].direction) {
      return false;
    }
  }
  return true;
}

bool configurations_are_equal(const ContinuousConfiguration &left,
                              const ContinuousConfiguration &right) {
  if (left.cycles != right.cycles || left.permutation != right.permutation ||
      left.worldlines.size() != right.worldlines.size() || left.log_Z0_N != right.log_Z0_N) {
    return false;
  }
  for (std::size_t index = 0; index < left.worldlines.size(); ++index) {
    if (!paths_are_equal(left.worldlines[index], right.worldlines[index])) {
      return false;
    }
  }
  return true;
}

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

TEST(ContinuousConfigurationTest, ReusableEnsembleMatchesOneOffWrapperForTheSameSeed) {
  const Model model{
      .particle_count = 6,
      .beta = 0.9,
      .linear_size = 5,
      .dimension = 2,
      .hopping = 0.7,
  };
  const CanonicalEnsemble ensemble(model);
  Random one_off_random(1907);
  Random retained_random(1907);

  const auto one_off = sample_ideal_continuous_configuration(model, one_off_random);
  const auto retained = sample_ideal_continuous_configuration(ensemble, retained_random);

  EXPECT_TRUE(configurations_are_equal(one_off, retained));
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

TEST(ContinuousConfigurationTest, TimeOriginRotationPreservesPhysicalInvariants) {
  const Model model{
      .particle_count = 7,
      .beta = 1.3,
      .linear_size = 6,
      .dimension = 2,
      .hopping = 0.8,
  };
  Random random(122);
  const auto state = sample_ideal_continuous_configuration(model, random);
  const double overlap = pair_overlap_time(state, model);
  const auto winding = state.total_winding(model);
  const auto rotated = rotate_configuration_time_origin(state, model, 0.43);
  EXPECT_NO_THROW(rotated.validate(model));
  EXPECT_EQ(rotated.event_count(), state.event_count());
  EXPECT_EQ(rotated.total_winding(model), winding);
  EXPECT_NEAR(pair_overlap_time(rotated, model), overlap, 2e-11);
}

TEST(ContinuousConfigurationTest, TimeOriginRotationRetainsJumpAtNewSeam) {
  const Model model{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  };
  const ContinuousPath path{
      .duration = 1.0,
      .start = {0},
      .end = {0},
      .events = {{.time = 0.25, .axis = 0, .direction = 1},
                 {.time = 0.75, .axis = 0, .direction = -1}},
  };
  const ContinuousConfiguration state{
      .cycles = {{0}},
      .permutation = {0},
      .worldlines = {path},
      .log_Z0_N = 0.0,
  };
  state.validate(model);
  const auto rotated = rotate_configuration_time_origin(state, model, 0.25);
  rotated.validate(model);
  ASSERT_EQ(rotated.worldlines[0].events.size(), 2U);
  EXPECT_DOUBLE_EQ(rotated.worldlines[0].events[0].time, 0.0);
  EXPECT_EQ(rotated.worldlines[0].events[0].direction, 1);
  EXPECT_EQ(rotated.worldlines[0].position_at(0.0), Site({1}));
  EXPECT_EQ(rotated.total_winding(model), state.total_winding(model));
}

} // namespace
} // namespace qmc
