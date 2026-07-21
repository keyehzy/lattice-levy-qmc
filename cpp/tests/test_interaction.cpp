#include "qmc/interaction.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace qmc {
namespace {

ContinuousPath static_path(const Coord site, const double beta = 1.0) {
  return ContinuousPath(beta, {site}, {site}, {});
}

ContinuousConfiguration two_particle_state(const Model &model, const ContinuousPath &first,
                                           const ContinuousPath &second) {
  return ContinuousConfiguration(model, Permutation({0, 1}), {first, second});
}

TEST(InteractionTest, IntegratesPiecewisePairOverlapExactly) {
  const Model free{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  };
  const InteractingModel model{.free = free, .interaction = 3.0};
  const ContinuousPath moving(
      1.0, {0}, {0},
      {{.time = 0.25, .axis = 0, .direction = 1}, {.time = 0.75, .axis = 0, .direction = -1}});
  const auto state = two_particle_state(free, static_path(0), moving);
  state.validate();

  EXPECT_DOUBLE_EQ(pair_overlap_time(state), 0.5);
  EXPECT_DOUBLE_EQ(interaction_action(state, model), 1.5);
  EXPECT_DOUBLE_EQ(kinetic_energy_estimator(state), -2.0);
  EXPECT_DOUBLE_EQ(interaction_energy_estimator(state, model), 1.5);
  EXPECT_DOUBLE_EQ(total_energy_estimator(state, model), -0.5);
  EXPECT_NEAR(double_occupancy_per_site(state), 1.0 / 6.0, 1e-15);
  EXPECT_DOUBLE_EQ(interaction_action(state, model, 0.2), 1.1);
}

TEST(InteractionTest, RejectsInteractingModelWithDifferentConfigurationProvenance) {
  const Model free{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  };
  const auto state = two_particle_state(free, static_path(0), static_path(1));
  InteractingModel mismatched{.free = free, .interaction = 2.0};
  mismatched.free.hopping = 0.5;

  EXPECT_THROW(static_cast<void>(interaction_action(state, mismatched)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(interaction_energy_estimator(state, mismatched)),
               std::invalid_argument);
}

TEST(InteractionTest, UsesEuclideanModuloForNegativeCoveringCoordinates) {
  const Model model{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  };
  const ContinuousPath moving(
      1.0, {-3}, {-3},
      {{.time = 0.25, .axis = 0, .direction = 1}, {.time = 0.75, .axis = 0, .direction = -1}});
  const auto state = two_particle_state(model, static_path(0), moving);
  EXPECT_DOUBLE_EQ(pair_overlap_time(state), 0.5);
}

TEST(InteractionTest, SimultaneousEventsHaveNoOrderingDuration) {
  const Model model{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 1.0,
  };
  const ContinuousPath first(
      1.0, {0}, {0},
      {{.time = 0.5, .axis = 0, .direction = 1}, {.time = 0.5, .axis = 0, .direction = -1}});
  const ContinuousPath second(
      1.0, {1}, {1},
      {{.time = 0.5, .axis = 0, .direction = -1}, {.time = 0.5, .axis = 0, .direction = 1}});
  const auto state = two_particle_state(model, first, second);
  EXPECT_DOUBLE_EQ(pair_overlap_time(state), 0.0);
}

TEST(InteractionTest, HandlesEmptyAndSingleParticleStates) {
  const Model empty_model{
      .particle_count = 0,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  };
  const ContinuousConfiguration empty(empty_model, Permutation(), {});
  EXPECT_DOUBLE_EQ(pair_overlap_time(empty), 0.0);

  const Model single_model{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  };
  const ContinuousConfiguration single(single_model, Permutation({0}), {static_path(0)});
  EXPECT_DOUBLE_EQ(pair_overlap_time(single), 0.0);
}

TEST(InteractingModelTest, SeparatesFreeAndInteractingValidation) {
  InteractingModel model{
      .free = {.particle_count = 2, .beta = 1.0, .linear_size = 3, .dimension = 1, .hopping = 1.0},
      .interaction = -2.0,
  };
  EXPECT_NO_THROW(model.validate());
  EXPECT_EQ(model.volume(), 3U);
  model.free.beta = 0.0;
  EXPECT_THROW(model.validate(), std::invalid_argument);
  model.free.beta = 1.0;
  model.interaction = std::numeric_limits<double>::infinity();
  EXPECT_THROW(model.validate(), std::invalid_argument);
}

} // namespace
} // namespace qmc
