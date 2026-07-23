#include "continuous_event_sweep.hpp"
#include "continuous_test_fixtures.hpp"
#include "interaction_detail.hpp"
#include "qmc/interaction.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

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
  const Model free(qmc::ModelParameters{
      .particle_count = 2,
      .beta = 2.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const InteractingModel model{.free = free, .interaction = 3.0};
  const ContinuousPath moving(
      2.0, {0}, {0},
      {{.time = 0.5, .axis = 0, .direction = 1}, {.time = 1.5, .axis = 0, .direction = -1}});
  const auto state = two_particle_state(free, static_path(0, 2.0), moving);
  state.validate();

  const InteractionMeasurement measurement = measure_interaction(state, model);
  EXPECT_DOUBLE_EQ(measurement.action, 3.0);
  EXPECT_DOUBLE_EQ(measurement.pair_overlap_time, 1.0);
  EXPECT_NEAR(measurement.double_occupancy_per_site, 1.0 / 6.0, 1e-15);
  EXPECT_DOUBLE_EQ(measurement.kinetic_energy, -1.0);
  EXPECT_DOUBLE_EQ(measurement.interaction_energy, 1.5);
  EXPECT_DOUBLE_EQ(measurement.total_energy, 0.5);
  EXPECT_EQ(measurement.event_count, 2U);
  EXPECT_DOUBLE_EQ(pair_overlap_time(state), 1.0);
  EXPECT_DOUBLE_EQ(interaction_action(state, model), 3.0);
  EXPECT_DOUBLE_EQ(kinetic_energy_estimator(state), -1.0);
  EXPECT_DOUBLE_EQ(interaction_energy_estimator(state, model), 1.5);
  EXPECT_DOUBLE_EQ(total_energy_estimator(state, model), 0.5);
  EXPECT_NEAR(double_occupancy_per_site(state), 1.0 / 6.0, 1e-15);
  EXPECT_DOUBLE_EQ(interaction_action(state, model, 0.2), 2.2);
}

TEST(InteractionTest, RejectsInteractingModelWithDifferentConfigurationProvenance) {
  const Model free(qmc::ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const auto state = two_particle_state(free, static_path(0), static_path(1));
  const InteractingModel mismatched{
      .free = Model(qmc::ModelParameters{
          .particle_count = free.particle_count(),
          .beta = free.beta(),
          .linear_size = free.linear_size(),
          .dimension = free.dimension(),
          .hopping = 0.5,
      }),
      .interaction = 2.0,
  };

  EXPECT_THROW(static_cast<void>(measure_interaction(state, mismatched)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(interaction_action(state, mismatched)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(interaction_energy_estimator(state, mismatched)),
               std::invalid_argument);
}

TEST(InteractionTest, UsesEuclideanModuloForNegativeCoveringCoordinates) {
  const Model model(qmc::ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousPath moving(
      1.0, {-3}, {-3},
      {{.time = 0.25, .axis = 0, .direction = 1}, {.time = 0.75, .axis = 0, .direction = -1}});
  const auto state = two_particle_state(model, static_path(0), moving);
  EXPECT_DOUBLE_EQ(pair_overlap_time(state), 0.5);
}

TEST(InteractionTest, SimultaneousEventsHaveNoOrderingDuration) {
  const Model model(qmc::ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousPath first(
      1.0, {0}, {0},
      {{.time = 0.5, .axis = 0, .direction = 1}, {.time = 0.5, .axis = 0, .direction = -1}});
  const ContinuousPath second(
      1.0, {1}, {1},
      {{.time = 0.5, .axis = 0, .direction = -1}, {.time = 0.5, .axis = 0, .direction = 1}});
  const auto state = two_particle_state(model, first, second);
  EXPECT_DOUBLE_EQ(pair_overlap_time(state), 0.0);
}

TEST(InteractionTest, SharedSweepPreservesCoincidentSeamAndTieSemantics) {
  const ContinuousConfiguration state = test::coincident_seam_configuration();
  const TorusLayout layout(state.model().linear_size(), state.model().dimension());

  const detail::ContinuousEventSweepData configuration_sweep =
      detail::build_continuous_event_sweep(state, layout);
  std::vector<const ContinuousPath *> path_view;
  path_view.reserve(state.worldlines().size());
  for (const ContinuousPath &path : state.worldlines()) {
    path_view.push_back(&path);
  }
  const detail::ContinuousEventSweepData path_view_sweep =
      detail::build_continuous_event_sweep(state.model(), path_view, layout);

  EXPECT_EQ(path_view_sweep.seam_positions, configuration_sweep.seam_positions);
  EXPECT_EQ(path_view_sweep.hops, configuration_sweep.hops);
  EXPECT_EQ(path_view_sweep.group_offsets, configuration_sweep.group_offsets);
  EXPECT_DOUBLE_EQ(detail::pair_overlap_time_for_paths(state.model(), path_view), 0.5);
  EXPECT_DOUBLE_EQ(pair_overlap_time(state), 0.5);
  const InteractionMeasurement measurement =
      measure_interaction(state, InteractingModel{.free = state.model(), .interaction = 2.0});
  EXPECT_DOUBLE_EQ(measurement.action, 1.0);

  for (const double shift : std::array{0.25, 0.5, 0.75}) {
    const ContinuousConfiguration rotated = rotate_configuration_time_origin(state, shift);
    EXPECT_DOUBLE_EQ(pair_overlap_time(rotated), 0.5);
  }
}

TEST(InteractionTest, HandlesEmptyAndSingleParticleStates) {
  const Model empty_model(qmc::ModelParameters{
      .particle_count = 0,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousConfiguration empty(empty_model, Permutation(), {});
  const auto empty_measurement =
      measure_interaction(empty, InteractingModel{.free = empty_model, .interaction = 2.0});
  EXPECT_DOUBLE_EQ(pair_overlap_time(empty), 0.0);
  EXPECT_DOUBLE_EQ(empty_measurement.action, 0.0);
  EXPECT_DOUBLE_EQ(empty_measurement.pair_overlap_time, 0.0);
  EXPECT_DOUBLE_EQ(empty_measurement.double_occupancy_per_site, 0.0);
  EXPECT_DOUBLE_EQ(empty_measurement.kinetic_energy, 0.0);
  EXPECT_DOUBLE_EQ(empty_measurement.interaction_energy, 0.0);
  EXPECT_DOUBLE_EQ(empty_measurement.total_energy, 0.0);
  EXPECT_EQ(empty_measurement.event_count, 0U);

  const Model single_model(qmc::ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousConfiguration single(single_model, Permutation({0}), {static_path(0)});
  const auto single_measurement =
      measure_interaction(single, InteractingModel{.free = single_model, .interaction = -2.0});
  EXPECT_DOUBLE_EQ(pair_overlap_time(single), 0.0);
  EXPECT_DOUBLE_EQ(single_measurement.action, 0.0);
  EXPECT_DOUBLE_EQ(single_measurement.pair_overlap_time, 0.0);
  EXPECT_DOUBLE_EQ(single_measurement.double_occupancy_per_site, 0.0);
  EXPECT_DOUBLE_EQ(single_measurement.kinetic_energy, 0.0);
  EXPECT_DOUBLE_EQ(single_measurement.interaction_energy, 0.0);
  EXPECT_DOUBLE_EQ(single_measurement.total_energy, 0.0);
  EXPECT_EQ(single_measurement.event_count, 0U);
}

TEST(InteractingModelTest, SeparatesFreeAndInteractingValidation) {
  InteractingModel model{
      .free = qmc::Model(qmc::ModelParameters{
          .particle_count = 2, .beta = 1.0, .linear_size = 3, .dimension = 1, .hopping = 1.0}),
      .interaction = -2.0,
  };
  EXPECT_NO_THROW(model.validate());
  EXPECT_EQ(model.volume(), 3U);
  model.free = Model(qmc::ModelParameters{
      .particle_count = 2,
      .beta = 0.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  EXPECT_THROW(model.validate(), std::invalid_argument);
  model.free = Model(qmc::ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  model.interaction = std::numeric_limits<double>::infinity();
  EXPECT_THROW(model.validate(), std::invalid_argument);
}

} // namespace
} // namespace qmc
