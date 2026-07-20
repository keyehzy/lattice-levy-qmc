#include "qmc/continuous_configuration.hpp"
#include "qmc/interaction.hpp"

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>

namespace qmc {
namespace {

bool paths_are_equal(const ContinuousPath &left, const ContinuousPath &right) {
  return left == right;
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

ContinuousConfiguration
reference_rotate_configuration_time_origin(const ContinuousConfiguration &state, const Model &model,
                                           const double shift) {
  std::vector<ContinuousPath> paths;
  paths.reserve(state.worldlines.size());
  for (std::size_t particle = 0; particle < state.worldlines.size(); ++particle) {
    const ContinuousPath &path = state.worldlines[particle];
    const ContinuousPath &successor = state.worldlines[state.permutation[particle]];
    const auto path_cut = std::ranges::lower_bound(path.events(), shift, {}, &JumpEvent::time);
    const auto successor_cut =
        std::ranges::lower_bound(successor.events(), shift, {}, &JumpEvent::time);

    Site start = path.start();
    for (auto event = path.events().begin(); event != path_cut; ++event) {
      start[event->axis] += static_cast<Coord>(event->direction);
    }

    std::vector<JumpEvent> events;
    for (auto event = path_cut; event != path.events().end(); ++event) {
      events.push_back(JumpEvent{
          .time = event->time - shift,
          .axis = event->axis,
          .direction = event->direction,
      });
    }
    for (auto event = successor.events().begin(); event != successor_cut; ++event) {
      events.push_back(JumpEvent{
          .time = (model.beta - shift) + event->time,
          .axis = event->axis,
          .direction = event->direction,
      });
    }

    Site end = successor.start();
    for (auto event = successor.events().begin(); event != successor_cut; ++event) {
      end[event->axis] += static_cast<Coord>(event->direction);
    }
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      end[axis] += path.end()[axis] - successor.start()[axis];
    }
    paths.emplace_back(model.beta, std::move(start), std::move(end), std::move(events));
  }
  return ContinuousConfiguration{
      .cycles = state.cycles,
      .permutation = state.permutation,
      .worldlines = std::move(paths),
      .log_Z0_N = state.log_Z0_N,
  };
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
    EXPECT_DOUBLE_EQ(path.duration(), model.beta);
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
  const ContinuousPath &path = state.worldlines[0];
  Site disconnected_end = path.end();
  ++disconnected_end[0];
  std::vector<JumpEvent> disconnected_events(path.events().begin(), path.events().end());
  disconnected_events.push_back(JumpEvent{.time = path.duration(), .axis = 0, .direction = 1});
  state.worldlines[0] = ContinuousPath(path.duration(), path.start(), std::move(disconnected_end),
                                       std::move(disconnected_events));
  EXPECT_THROW(state.validate(model), std::logic_error);
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
  const auto expected = reference_rotate_configuration_time_origin(state, model, 0.43);
  const auto rotated = rotate_configuration_time_origin(state, model, 0.43);
  EXPECT_TRUE(configurations_are_equal(rotated, expected));
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
  const ContinuousPath path(
      1.0, {0}, {0},
      {{.time = 0.25, .axis = 0, .direction = 1}, {.time = 0.75, .axis = 0, .direction = -1}});
  const ContinuousConfiguration state{
      .cycles = {{0}},
      .permutation = {0},
      .worldlines = {path},
      .log_Z0_N = 0.0,
  };
  state.validate(model);
  const auto rotated = rotate_configuration_time_origin(state, model, 0.25);
  rotated.validate(model);
  ASSERT_EQ(rotated.worldlines[0].events().size(), 2U);
  EXPECT_DOUBLE_EQ(rotated.worldlines[0].events()[0].time, 0.0);
  EXPECT_EQ(rotated.worldlines[0].events()[0].direction, 1);
  EXPECT_EQ(rotated.worldlines[0].position_at(0.0), Site({1}));
  EXPECT_EQ(rotated.total_winding(model), state.total_winding(model));
}

TEST(ContinuousConfigurationTest, CursorRotationMatchesCoincidentSeamTraversalExactly) {
  const Model model{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 5,
      .dimension = 1,
      .hopping = 1.0,
  };
  const ContinuousPath first(1.0, {0}, {1},
                             {{.time = 0.0, .axis = 0, .direction = 1},
                              {.time = 0.25, .axis = 0, .direction = 1},
                              {.time = 0.25, .axis = 0, .direction = -1},
                              {.time = 0.75, .axis = 0, .direction = 1},
                              {.time = 1.0, .axis = 0, .direction = -1}});
  const ContinuousPath second(1.0, {1}, {0},
                              {{.time = 0.0, .axis = 0, .direction = -1},
                               {.time = 0.25, .axis = 0, .direction = 1},
                               {.time = 0.5, .axis = 0, .direction = 1},
                               {.time = 0.75, .axis = 0, .direction = -1},
                               {.time = 0.75, .axis = 0, .direction = 1},
                               {.time = 1.0, .axis = 0, .direction = -1},
                               {.time = 1.0, .axis = 0, .direction = -1}});
  const ContinuousConfiguration state{
      .cycles = {{0, 1}},
      .permutation = {1, 0},
      .worldlines = {first, second},
      .log_Z0_N = 0.0,
  };
  state.validate(model);

  const ContinuousConfiguration expected =
      reference_rotate_configuration_time_origin(state, model, 0.25);
  const ContinuousConfiguration actual = rotate_configuration_time_origin(state, model, 0.25);
  EXPECT_TRUE(configurations_are_equal(actual, expected));
  EXPECT_NO_THROW(actual.validate(model));
  EXPECT_EQ(actual.event_count(), state.event_count());
  ASSERT_GE(actual.worldlines[0].events().size(), 2U);
  EXPECT_DOUBLE_EQ(actual.worldlines[0].events()[0].time, 0.0);
  EXPECT_DOUBLE_EQ(actual.worldlines[0].events()[1].time, 0.0);
}

TEST(ContinuousConfigurationTest, CursorRotationPreservesAllowedDurationDriftSemantics) {
  const Model model{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  };
  const double one_ulp_below_beta = std::nextafter(model.beta, 0.0);
  const double duration = std::nextafter(one_ulp_below_beta, 0.0);
  const ContinuousConfiguration state{
      .cycles = {{0}},
      .permutation = {0},
      .worldlines = {ContinuousPath(duration, {0}, {0},
                                    {{.time = duration, .axis = 0, .direction = 1},
                                     {.time = duration, .axis = 0, .direction = -1}})},
      .log_Z0_N = 0.0,
  };
  state.validate(model);

  const ContinuousConfiguration expected =
      reference_rotate_configuration_time_origin(state, model, one_ulp_below_beta);
  const ContinuousConfiguration actual =
      rotate_configuration_time_origin(state, model, one_ulp_below_beta);
  EXPECT_TRUE(configurations_are_equal(actual, expected));
  EXPECT_NO_THROW(actual.validate(model));
}

} // namespace
} // namespace qmc
