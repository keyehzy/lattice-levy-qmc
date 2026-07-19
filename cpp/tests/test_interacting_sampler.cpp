#include "qmc/interacting_sampler.hpp"
#include "qmc/interaction.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <optional>

namespace qmc {
namespace {

InteractingModel sampler_model(const double interaction) {
  return InteractingModel{
      .free = {.particle_count = 6, .beta = 1.1, .linear_size = 5, .dimension = 2, .hopping = 0.8},
      .interaction = interaction,
  };
}

bool equal_path(const ContinuousPath &left, const ContinuousPath &right) {
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

bool equal_state(const ContinuousConfiguration &left, const ContinuousConfiguration &right) {
  if (left.cycles != right.cycles || left.permutation != right.permutation ||
      left.worldlines.size() != right.worldlines.size() || left.log_Z0_N != right.log_Z0_N) {
    return false;
  }
  for (std::size_t index = 0; index < left.worldlines.size(); ++index) {
    if (!equal_path(left.worldlines[index], right.worldlines[index])) {
      return false;
    }
  }
  return true;
}

TEST(InteractingSamplerTest, ZeroInteractionMovesAreRejectionFree) {
  InteractingSampler sampler(sampler_model(0.0), 12);
  for (int update = 0; update < 20; ++update) {
    EXPECT_TRUE(sampler.segment_update(std::nullopt, std::nullopt, 0.4));
    EXPECT_TRUE(sampler.cycle_update());
    EXPECT_TRUE(sampler.global_update());
  }
  for (const MoveStatistics &statistics : sampler.statistics()) {
    ASSERT_TRUE(statistics.acceptance().has_value());
    EXPECT_DOUBLE_EQ(*statistics.acceptance(), 1.0);
  }
}

TEST(InteractingSamplerTest, FiniteInteractionPreservesStateAndCacheInvariants) {
  InteractingSampler sampler(sampler_model(2.0), 13);
  const SweepOptions options{
      .segment_updates = 6,
      .segment_fraction = 0.35,
      .cycle_updates = 2,
      .global_updates = 1,
  };
  for (int sweep = 0; sweep < 20; ++sweep) {
    sampler.sweep(options);
    EXPECT_NO_THROW(sampler.state().validate(sampler.model().free));
    const double recomputed = pair_overlap_time(sampler.state(), sampler.model().free);
    EXPECT_NEAR(sampler.pair_overlap(), recomputed, 1e-12);
    EXPECT_NEAR(sampler.action(), sampler.model().interaction * recomputed, 1e-12);
    EXPECT_GE(sampler.pair_overlap(), 0.0);
  }

  const auto value = sampler.observables();
  EXPECT_DOUBLE_EQ(value.action, sampler.action());
  EXPECT_DOUBLE_EQ(value.pair_overlap_time, sampler.pair_overlap());
  EXPECT_EQ(value.event_count, sampler.state().event_count());
  EXPECT_EQ(value.winding, sampler.state().total_winding(sampler.model().free));
  EXPECT_EQ(value.cycle_lengths, sampler.state().cycle_lengths());
  EXPECT_NEAR(value.total_energy, value.kinetic_energy + value.interaction_energy, 1e-14);
}

TEST(InteractingSamplerTest, RejectedMovesLeaveAcceptedStateUntouched) {
  InteractingModel model{
      .free = {.particle_count = 5, .beta = 2.0, .linear_size = 2, .dimension = 1, .hopping = 1.0},
      .interaction = 100.0,
  };
  InteractingSampler sampler(model, 927);
  bool observed_rejection = false;
  for (int attempt = 0; attempt < 200 && !observed_rejection; ++attempt) {
    const ContinuousConfiguration before = sampler.state();
    const double overlap_before = sampler.pair_overlap();
    const double action_before = sampler.action();
    if (!sampler.global_update()) {
      observed_rejection = true;
      EXPECT_TRUE(equal_state(sampler.state(), before));
      EXPECT_DOUBLE_EQ(sampler.pair_overlap(), overlap_before);
      EXPECT_DOUBLE_EQ(sampler.action(), action_before);
    }
  }
  EXPECT_TRUE(observed_rejection);
}

TEST(InteractingSamplerTest, RunAppliesBurnInThinningAndReturnsTypedMeasurements) {
  InteractingSampler sampler(sampler_model(1.0), 77);
  const RunOptions options{
      .burn_in = 2,
      .thin = 3,
      .sweep = {.segment_updates = 2,
                .segment_fraction = 0.3,
                .cycle_updates = 1,
                .global_updates = 1},
  };
  const auto samples = sampler.run(5, options);
  ASSERT_EQ(samples.size(), 5U);
  EXPECT_EQ(sampler.statistics(MoveKind::SegmentMove).attempts, 34U);
  EXPECT_EQ(sampler.statistics(MoveKind::CycleMove).attempts, 17U);
  EXPECT_EQ(sampler.statistics(MoveKind::GlobalMove).attempts, 17U);
  EXPECT_THROW(static_cast<void>(sampler.run(1, RunOptions{.burn_in = 0, .thin = 0, .sweep = {}})),
               std::invalid_argument);
}

TEST(InteractingSamplerTest, EmptySystemLocalMovesAreNoOps) {
  const InteractingModel model{
      .free = {.particle_count = 0, .beta = 1.0, .linear_size = 3, .dimension = 1, .hopping = 1.0},
      .interaction = 2.0,
  };
  InteractingSampler sampler(model, 5);
  EXPECT_TRUE(sampler.segment_update(99, std::nullopt, -1.0));
  EXPECT_TRUE(sampler.cycle_update(99));
  EXPECT_EQ(sampler.statistics(MoveKind::SegmentMove).attempts, 0U);
  EXPECT_EQ(sampler.statistics(MoveKind::CycleMove).attempts, 0U);
  EXPECT_TRUE(sampler.global_update());
  EXPECT_EQ(sampler.statistics(MoveKind::GlobalMove).attempts, 1U);
  EXPECT_EQ(sampler.statistics(MoveKind::GlobalMove).accepts, 1U);
}

TEST(InteractingSamplerTest, ValidatesExplicitMoveArguments) {
  InteractingSampler sampler(sampler_model(1.0), 91);
  EXPECT_THROW(static_cast<void>(sampler.segment_update(99)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sampler.segment_update(0, std::nullopt, 0.0)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sampler.segment_update(0, std::pair<double, double>{0.8, 0.2})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sampler.cycle_update(99)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sampler.statistics(static_cast<MoveKind>(99))),
               std::invalid_argument);
}

TEST(InteractingSamplerTest, SupportsAttractiveInteractionAndZeroHopping) {
  const InteractingModel model{
      .free = {.particle_count = 3, .beta = 1.0, .linear_size = 2, .dimension = 1, .hopping = 0.0},
      .interaction = -2.0,
  };
  InteractingSampler sampler(model, 123);
  for (int update = 0; update < 10; ++update) {
    sampler.sweep(SweepOptions{
        .segment_updates = 3, .segment_fraction = 0.5, .cycle_updates = 1, .global_updates = 1});
  }
  EXPECT_EQ(sampler.state().event_count(), 0U);
  EXPECT_NO_THROW(sampler.state().validate(model.free));
  EXPECT_NEAR(sampler.action(), model.interaction * sampler.pair_overlap(), 1e-14);
}

} // namespace
} // namespace qmc
