#include "continuous_test_fixtures.hpp"
#include "qmc/continuous_configuration.hpp"
#include "qmc/interaction.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace qmc {
namespace {

static_assert(!std::default_initializable<ContinuousConfiguration>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousConfiguration &>().worldlines()),
                             std::span<const ContinuousPath>>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousConfiguration &>().model()),
                             const Model &>);

ContinuousConfiguration
reference_rotate_configuration_time_origin(const ContinuousConfiguration &state, const Model &model,
                                           const double shift) {
  std::vector<ContinuousPath> paths;
  paths.reserve(state.worldlines().size());
  for (std::size_t particle = 0; particle < state.worldlines().size(); ++particle) {
    const ContinuousPath &path = state.path(static_cast<ParticleId>(particle));
    const ContinuousPath &successor =
        state.path(state.topology().successor(static_cast<ParticleId>(particle)));
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
          .time = (model.beta() - shift) + event->time,
          .axis = event->axis,
          .direction = event->direction,
      });
    }

    Site end = successor.start();
    for (auto event = successor.events().begin(); event != successor_cut; ++event) {
      end[event->axis] += static_cast<Coord>(event->direction);
    }
    for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
      end[axis] += path.end()[axis] - successor.start()[axis];
    }
    paths.emplace_back(model.beta(), std::move(start), std::move(end), std::move(events));
  }
  return ContinuousConfiguration(model, state.topology(), std::move(paths));
}

TEST(PermutationTest, ValidatesSuccessorsAndCachesDeterministicCycles) {
  const Permutation topology({2, 0, 1, 4, 3});
  EXPECT_EQ(topology.size(), 5U);
  EXPECT_EQ(topology.successor(0), 2U);
  EXPECT_EQ(topology.successor(4), 3U);
  EXPECT_TRUE(std::ranges::equal(topology.successors(), std::vector<ParticleId>{2, 0, 1, 4, 3}));
  ASSERT_EQ(topology.cycles().size(), 2U);
  EXPECT_EQ(topology.cycles()[0], (Cycle{0, 2, 1}));
  EXPECT_EQ(topology.cycles()[1], (Cycle{3, 4}));
}

TEST(PermutationTest, RejectsMalformedSuccessorsAtConstruction) {
  EXPECT_THROW(static_cast<void>(Permutation({0, 0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Permutation({1})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Permutation({1, 2, 3})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Permutation({0}).successor(1)), std::out_of_range);
}

TEST(ContinuousConfigurationTest, SamplesConsistentCanonicalState) {
  const Model model(qmc::ModelParameters{
      .particle_count = 8,
      .beta = 1.3,
      .linear_size = 7,
      .dimension = 2,
      .hopping = 0.75,
  });
  Random random(11);
  const auto state = sample_ideal_continuous_configuration(model, random);
  EXPECT_NO_THROW(state.validate());
  EXPECT_EQ(state.model(), model);
  EXPECT_EQ(state.topology().size(), 8U);
  EXPECT_EQ(state.worldlines().size(), 8U);
  const auto lengths = state.cycle_lengths();
  EXPECT_EQ(std::accumulate(lengths.begin(), lengths.end(), std::size_t{0}), 8U);
  EXPECT_EQ(state.total_winding().size(), 2U);
  for (const ContinuousPath &path : state.worldlines()) {
    EXPECT_DOUBLE_EQ(path.duration(), model.beta());
  }
  for (const Site &position : state.positions_at(0.7)) {
    ASSERT_EQ(position.size(), 2U);
    EXPECT_TRUE(std::ranges::all_of(
        position, [](const Coord coordinate) { return coordinate >= 0 && coordinate < 7; }));
  }
}

TEST(ContinuousConfigurationTest, NormalizesLongCycleCutDriftForNonDyadicBeta) {
  const Model model(qmc::ModelParameters{
      .particle_count = 64,
      .beta = 2.0 / 3.0,
      .linear_size = 8,
      .dimension = 2,
      .hopping = 1.0,
  });
  Random random(23003);

  const auto state = sample_ideal_continuous_configuration(model, random);

  EXPECT_NO_THROW(state.validate());
  for (const ContinuousPath &path : state.worldlines()) {
    EXPECT_DOUBLE_EQ(path.duration(), model.beta());
  }
}

TEST(ContinuousConfigurationTest, SupportsEmptyCanonicalState) {
  const Model model(qmc::ModelParameters{
      .particle_count = 0,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 2,
      .hopping = 1.0,
  });
  Random random(3);
  const auto state = sample_ideal_continuous_configuration(model, random);
  EXPECT_NO_THROW(state.validate());
  EXPECT_TRUE(state.topology().cycles().empty());
  EXPECT_TRUE(state.worldlines().empty());
  EXPECT_EQ(state.event_count(), 0U);
  EXPECT_EQ(state.total_winding(), Site({0, 0}));
}

TEST(ContinuousConfigurationTest, OwnsModelProvenanceAndBoundsPathAccess) {
  Model model(qmc::ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 0.75,
  });
  const ContinuousConfiguration state(model, Permutation({0}), {ContinuousPath(1.0, {0}, {0}, {})});
  model = Model(qmc::ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 9.0,
  });

  EXPECT_DOUBLE_EQ(state.model().hopping(), 0.75);
  EXPECT_EQ(state.path(0), ContinuousPath(1.0, {0}, {0}, {}));
  EXPECT_THROW(static_cast<void>(state.path(1)), std::out_of_range);
}

TEST(ContinuousConfigurationTest, ReusableEnsembleMatchesOneOffWrapperForTheSameSeed) {
  const Model model(qmc::ModelParameters{
      .particle_count = 6,
      .beta = 0.9,
      .linear_size = 5,
      .dimension = 2,
      .hopping = 0.7,
  });
  const NumericalOptions numerical{
      .tail_tolerance = 1e-12,
      .max_bessel_terms = 50'000,
      .max_winding = 20'000,
  };
  const CanonicalEnsemble ensemble(model, numerical);
  const CanonicalEnsemble default_ensemble(model);
  Random one_off_random(1907);
  Random retained_random(1907);
  Random compatibility_random(1907);

  const auto one_off = sample_ideal_continuous_configuration(model, one_off_random, numerical);
  const auto retained = sample_ideal_continuous_configuration(ensemble, retained_random);
  const auto compatibility =
      sample_ideal_continuous_configuration(default_ensemble, compatibility_random, numerical);

  EXPECT_EQ(one_off, retained);
  EXPECT_EQ(retained, compatibility);
  const double one_off_next = one_off_random.uniform_open();
  const double retained_next = retained_random.uniform_open();
  const double compatibility_next = compatibility_random.uniform_open();
  EXPECT_DOUBLE_EQ(one_off_next, retained_next);
  EXPECT_DOUBLE_EQ(retained_next, compatibility_next);
}

TEST(ContinuousConfigurationTest, ConstructionRejectsInvalidShapeDurationAndConnectivity) {
  const Model model(qmc::ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 5,
      .dimension = 1,
      .hopping = 0.8,
  });
  const ContinuousPath valid(1.0, {0}, {0}, {});
  const Model zero_beta(qmc::ModelParameters{
      .particle_count = model.particle_count(),
      .beta = 0.0,
      .linear_size = model.linear_size(),
      .dimension = model.dimension(),
      .hopping = model.hopping(),
  });
  EXPECT_THROW(static_cast<void>(ContinuousConfiguration(zero_beta, Permutation({0}), {valid})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ContinuousConfiguration(model, Permutation(), {valid})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ContinuousConfiguration(model, Permutation({0}), {})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ContinuousConfiguration(model, Permutation({0}),
                                                         {ContinuousPath(0.5, {0}, {0}, {})})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ContinuousConfiguration(
                   model, Permutation({0}), {ContinuousPath(1.0, {0, 0}, {0, 0}, {})})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ContinuousConfiguration(
                   model, Permutation({0}),
                   {ContinuousPath(1.0, {0}, {1}, {{.time = 0.5, .axis = 0, .direction = 1}})})),
               std::invalid_argument);
}

TEST(ContinuousConfigurationTest, RejectsZeroBetaContinuousSampling) {
  const Model model(qmc::ModelParameters{
      .particle_count = 1,
      .beta = 0.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  Random random(9);
  EXPECT_THROW(static_cast<void>(sample_ideal_continuous_configuration(model, random)),
               std::invalid_argument);
}

TEST(ContinuousConfigurationTest, TimeOriginRotationPreservesPhysicalInvariants) {
  const Model model(qmc::ModelParameters{
      .particle_count = 7,
      .beta = 1.3,
      .linear_size = 6,
      .dimension = 2,
      .hopping = 0.8,
  });
  Random random(122);
  const auto state = sample_ideal_continuous_configuration(model, random);
  const double overlap = pair_overlap_time(state);
  const auto winding = state.total_winding();
  const auto expected = reference_rotate_configuration_time_origin(state, model, 0.43);
  const auto rotated = rotate_configuration_time_origin(state, 0.43);
  EXPECT_EQ(rotated, expected);
  EXPECT_NO_THROW(rotated.validate());
  EXPECT_EQ(rotated.event_count(), state.event_count());
  EXPECT_EQ(rotated.total_winding(), winding);
  EXPECT_NEAR(pair_overlap_time(rotated), overlap, 2e-11);
}

TEST(ContinuousConfigurationTest, TimeOriginRotationRetainsJumpAtNewSeam) {
  const Model model(qmc::ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousPath path(
      1.0, {0}, {0},
      {{.time = 0.25, .axis = 0, .direction = 1}, {.time = 0.75, .axis = 0, .direction = -1}});
  const ContinuousConfiguration state(model, Permutation({0}), {path});
  state.validate();
  const auto rotated = rotate_configuration_time_origin(state, 0.25);
  rotated.validate();
  ASSERT_EQ(rotated.path(0).events().size(), 2U);
  EXPECT_DOUBLE_EQ(rotated.path(0).events()[0].time, 0.0);
  EXPECT_EQ(rotated.path(0).events()[0].direction, 1);
  EXPECT_EQ(rotated.path(0).position_at(0.0), Site({1}));
  EXPECT_EQ(rotated.total_winding(), state.total_winding());
}

TEST(ContinuousConfigurationTest, CursorRotationMatchesCoincidentSeamTraversalExactly) {
  const ContinuousConfiguration state = test::coincident_seam_configuration();
  const Model &model = state.model();
  state.validate();

  const ContinuousConfiguration expected =
      reference_rotate_configuration_time_origin(state, model, 0.25);
  const ContinuousConfiguration actual = rotate_configuration_time_origin(state, 0.25);
  EXPECT_EQ(actual, expected);
  EXPECT_NO_THROW(actual.validate());
  EXPECT_EQ(actual.event_count(), state.event_count());
  ASSERT_GE(actual.path(0).events().size(), 2U);
  EXPECT_DOUBLE_EQ(actual.path(0).events()[0].time, 0.0);
  EXPECT_DOUBLE_EQ(actual.path(0).events()[1].time, 0.0);
}

TEST(ContinuousConfigurationTest, CursorRotationPreservesAllowedDurationDriftSemantics) {
  const Model model(qmc::ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const double one_ulp_below_beta = std::nextafter(model.beta(), 0.0);
  const double duration = std::nextafter(one_ulp_below_beta, 0.0);
  const ContinuousConfiguration state(
      model, Permutation({0}),
      {ContinuousPath(duration, {0}, {0},
                      {{.time = duration, .axis = 0, .direction = 1},
                       {.time = duration, .axis = 0, .direction = -1}})});
  state.validate();

  const ContinuousConfiguration expected =
      reference_rotate_configuration_time_origin(state, model, one_ulp_below_beta);
  const ContinuousConfiguration actual =
      rotate_configuration_time_origin(state, one_ulp_below_beta);
  EXPECT_EQ(actual, expected);
  EXPECT_NO_THROW(actual.validate());
}

} // namespace
} // namespace qmc
