#include "accepted_chain_state.hpp"
#include "qmc/interacting_sampler.hpp"
#include "qmc/interaction.hpp"
#include "stitch_matching.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace qmc {

namespace detail {

struct InteractingSamplerTestAccess {
  static bool occupancy_matches_state(const InteractingSampler &sampler) {
    return sampler.accepted_state_->occupancy_matches_configuration();
  }

  static double occupancy_overlap(InteractingSampler &sampler) {
    return sampler.accepted_state_->occupancy_pair_overlap();
  }

  static void set_interaction(InteractingSampler &sampler, const double interaction) {
    sampler.model_.interaction = interaction;
  }

  static void set_segment_accepts(InteractingSampler &sampler, const std::uint64_t accepts) {
    sampler.statistics_[static_cast<std::size_t>(MoveKind::SegmentMove)].accepts = accepts;
  }

  static bool try_invalid_topology_proposal(InteractingSampler &sampler) {
    std::vector<InteractingSampler::LabeledPath> replacements;
    replacements.emplace_back(0, sampler.state().worldlines[0]);
    std::vector<ParticleId> invalid_permutation(sampler.state().worldlines.size(), 0);
    return sampler.try_proposal(
        InteractingSampler::LocalProposal{
            .replacements = std::move(replacements),
            .successors = std::move(invalid_permutation),
            .successor_changes = 1,
        },
        sampler.statistics_[static_cast<std::size_t>(MoveKind::StitchMove)]);
  }
};

} // namespace detail

namespace {

InteractingModel sampler_model(const double interaction) {
  return InteractingModel{
      .free = {.particle_count = 6, .beta = 1.1, .linear_size = 5, .dimension = 2, .hopping = 0.8},
      .interaction = interaction,
  };
}

bool equal_path(const ContinuousPath &left, const ContinuousPath &right) { return left == right; }

bool equal_state(const ContinuousConfiguration &left, const ContinuousConfiguration &right) {
  if (left.topology() != right.topology() || left.worldlines.size() != right.worldlines.size() ||
      left.log_Z0_N != right.log_Z0_N) {
    return false;
  }
  for (std::size_t index = 0; index < left.worldlines.size(); ++index) {
    if (!equal_path(left.worldlines[index], right.worldlines[index])) {
      return false;
    }
  }
  return true;
}

struct BruteMatchingLaw {
  std::array<std::array<std::size_t, 3>, 6> matchings{};
  std::array<double, 6> weights{};
  double permanent = 0.0;
};

BruteMatchingLaw brute_matching_law(const std::array<double, 9> &weights) {
  BruteMatchingLaw law;
  std::array<std::size_t, 3> permutation{0, 1, 2};
  std::size_t index = 0;
  do {
    law.matchings[index] = permutation;
    law.weights[index] =
        weights[permutation[0]] * weights[3 + permutation[1]] * weights[6 + permutation[2]];
    law.permanent += law.weights[index];
    ++index;
  } while (std::ranges::next_permutation(permutation).found);
  return law;
}

std::size_t matching_rank(const std::vector<std::size_t> &matching, const BruteMatchingLaw &law) {
  for (std::size_t index = 0; index < law.matchings.size(); ++index) {
    if (std::ranges::equal(matching, law.matchings[index])) {
      return index;
    }
  }
  throw std::logic_error("sampled matching is not a permutation");
}

TEST(InteractingSamplerTest, PermanentRecursionMatchesBruteForceAndLimits) {
  const std::array<double, 9> weights{1.0, 2.0, 0.4, 0.7, 1.5, 3.0, 2.2, 0.8, 1.1};
  std::array<double, 9> log_weights{};
  std::ranges::transform(weights, log_weights.begin(),
                         [](const double value) { return std::log(value); });
  const std::vector<double> table = detail::log_permanent_table(log_weights, 3);
  const BruteMatchingLaw law = brute_matching_law(weights);
  EXPECT_NEAR(std::exp(table[0]), law.permanent, 2e-14);

  const std::array<double, 64> all_ones{};
  EXPECT_NEAR(std::exp(detail::log_permanent_table(all_ones, 8)[0]), 40'320.0, 1e-9);
  std::array<double, 64> unique{};
  unique.fill(-std::numeric_limits<double>::infinity());
  for (std::size_t index = 0; index < 8; ++index) {
    unique[(index * 8) + index] = 0.0;
  }
  const std::vector<double> unique_table = detail::log_permanent_table(unique, 8);
  Random random(72);
  EXPECT_EQ(detail::sample_permanent_matching(unique, 8, unique_table, random),
            (std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(InteractingSamplerTest, PermanentMatchingSamplerMatchesBruteForceLaw) {
  const std::array<double, 9> weights{1.0, 2.0, 0.4, 0.7, 1.5, 3.0, 2.2, 0.8, 1.1};
  std::array<double, 9> log_weights{};
  std::ranges::transform(weights, log_weights.begin(),
                         [](const double value) { return std::log(value); });
  const std::vector<double> table = detail::log_permanent_table(log_weights, 3);
  const BruteMatchingLaw law = brute_matching_law(weights);

  Random random(72);
  std::array<std::size_t, 6> counts{};
  for (std::size_t draw = 0; draw < 30'000; ++draw) {
    const std::vector<std::size_t> matching =
        detail::sample_permanent_matching(log_weights, 3, table, random);
    ++counts[matching_rank(matching, law)];
  }
  for (std::size_t index = 0; index < law.weights.size(); ++index) {
    EXPECT_NEAR(static_cast<double>(counts[index]) / 30'000.0, law.weights[index] / law.permanent,
                0.012);
  }
}

TEST(InteractingSamplerTest, ZeroInteractionMovesAreRejectionFree) {
  InteractingSampler sampler(sampler_model(0.0), 12);
  for (int update = 0; update < 20; ++update) {
    EXPECT_TRUE(sampler.segment_update(std::nullopt, std::nullopt, 0.4));
    EXPECT_TRUE(sampler.cycle_update());
    EXPECT_TRUE(sampler.stitch_update(std::nullopt, std::nullopt, std::nullopt, 0.3));
    EXPECT_TRUE(sampler.time_shift_update());
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
      .stitch_updates = 3,
      .stitch_fraction = 0.25,
      .stitch_mixture = {},
      .time_shift_updates = 1,
  };
  for (int sweep = 0; sweep < 20; ++sweep) {
    sampler.sweep(options);
    EXPECT_NO_THROW(sampler.state().validate(sampler.model().free));
    const double recomputed = pair_overlap_time(sampler.state(), sampler.model().free);
    EXPECT_NEAR(sampler.pair_overlap(), recomputed, 1e-12);
    EXPECT_NEAR(sampler.action(), sampler.model().interaction * recomputed, 1e-12);
    EXPECT_GE(sampler.pair_overlap(), 0.0);
    EXPECT_TRUE(detail::InteractingSamplerTestAccess::occupancy_matches_state(sampler));
    EXPECT_NEAR(detail::InteractingSamplerTestAccess::occupancy_overlap(sampler),
                sampler.pair_overlap(), 1e-12);
  }

  const auto value = sampler.observables();
  EXPECT_DOUBLE_EQ(value.action, sampler.action());
  EXPECT_DOUBLE_EQ(value.pair_overlap_time, sampler.pair_overlap());
  EXPECT_EQ(value.event_count, sampler.state().event_count());
  EXPECT_EQ(value.winding, sampler.state().total_winding(sampler.model().free));
  EXPECT_EQ(value.cycle_lengths, sampler.state().cycle_lengths());
  EXPECT_NEAR(value.total_energy, value.kinetic_energy + value.interaction_energy, 1e-14);
}

TEST(InteractingSamplerTest, ActionIsDerivedFromTheAcceptedOverlap) {
  InteractingSampler sampler(sampler_model(1.0), 131);
  const double overlap = sampler.pair_overlap();
  detail::InteractingSamplerTestAccess::set_interaction(sampler, -3.5);
  EXPECT_DOUBLE_EQ(sampler.action(), -3.5 * overlap);
  EXPECT_DOUBLE_EQ(sampler.observables().action, -3.5 * overlap);
  EXPECT_DOUBLE_EQ(sampler.pair_overlap(), overlap);
  EXPECT_TRUE(detail::InteractingSamplerTestAccess::occupancy_matches_state(sampler));
}

TEST(InteractingSamplerTest, RebuiltStatesTakeTheirOverlapFromTheOccupancyLedger) {
  InteractingSampler sampler(sampler_model(0.0), 1'311);
  EXPECT_DOUBLE_EQ(sampler.pair_overlap(),
                   detail::InteractingSamplerTestAccess::occupancy_overlap(sampler));

  EXPECT_TRUE(sampler.global_update());
  EXPECT_DOUBLE_EQ(sampler.pair_overlap(),
                   detail::InteractingSamplerTestAccess::occupancy_overlap(sampler));
  EXPECT_TRUE(detail::InteractingSamplerTestAccess::occupancy_matches_state(sampler));

  EXPECT_TRUE(sampler.time_shift_update(0.37));
  EXPECT_DOUBLE_EQ(sampler.pair_overlap(),
                   detail::InteractingSamplerTestAccess::occupancy_overlap(sampler));
  EXPECT_TRUE(detail::InteractingSamplerTestAccess::occupancy_matches_state(sampler));
}

TEST(InteractingSamplerTest, CopiesRetainAnAuthoritativeAcceptedState) {
  InteractingSampler original(sampler_model(1.7), 132);
  original.sweep(SweepOptions{.segment_updates = 4,
                              .segment_fraction = 0.4,
                              .cycle_updates = 1,
                              .global_updates = 1,
                              .stitch_updates = 2,
                              .stitch_mixture = {}});

  InteractingSampler copy(original);
  EXPECT_TRUE(equal_state(copy.state(), original.state()));
  EXPECT_DOUBLE_EQ(copy.pair_overlap(), original.pair_overlap());
  EXPECT_DOUBLE_EQ(copy.action(), original.action());
  EXPECT_TRUE(detail::InteractingSamplerTestAccess::occupancy_matches_state(copy));
  EXPECT_NEAR(detail::InteractingSamplerTestAccess::occupancy_overlap(copy), copy.pair_overlap(),
              1e-12);
}

TEST(InteractingSamplerTest, StitchUpdatesChangeTopologyAndKeepActionCacheExact) {
  const InteractingModel model{
      .free = {.particle_count = 8, .beta = 0.9, .linear_size = 9, .dimension = 1, .hopping = 1.0},
      .interaction = 1.4,
  };
  InteractingSampler sampler(model, 121);
  for (int step = 0; step < 200; ++step) {
    static_cast<void>(
        sampler.stitch_update(std::nullopt, std::nullopt, std::nullopt, 0.25, 2, 0.1));
    if (step % 10 == 0) {
      EXPECT_NO_THROW(sampler.state().validate(model.free));
      const double recomputed = pair_overlap_time(sampler.state(), model.free);
      EXPECT_NEAR(sampler.pair_overlap(), recomputed, 2e-11);
      EXPECT_NEAR(sampler.action(), model.interaction * recomputed, 2e-11);
    }
  }
  EXPECT_GT(sampler.statistics(MoveKind::StitchMove).topology_changes, 0U);
  const auto topology_rate = sampler.statistics(MoveKind::StitchMove).topology_change_rate();
  ASSERT_TRUE(topology_rate.has_value());
  EXPECT_GT(*topology_rate, 0.0);
}

TEST(InteractingSamplerTest, CollectiveStitchMixtureKeepsActionCacheExact) {
  const InteractingModel model{
      .free = {.particle_count = 8, .beta = 0.9, .linear_size = 7, .dimension = 1, .hopping = 1.2},
      .interaction = 2.3,
  };
  InteractingSampler sampler(model, 1231);
  const StitchMixture mixture{
      .strand_counts = {2, 3, 4},
      .strand_weights = {1.0, 1.0, 1.0},
  };
  for (int step = 0; step < 120; ++step) {
    sampler.stitch_sweep(1, 0.55, std::nullopt, 2, 0.1, mixture);
    if (step % 9 == 0) {
      EXPECT_NO_THROW(sampler.state().validate(model.free));
      const double recomputed = pair_overlap_time(sampler.state(), model.free);
      EXPECT_NEAR(sampler.pair_overlap(), recomputed, 2e-11);
      EXPECT_NEAR(sampler.action(), model.interaction * recomputed, 2e-11);
    }
  }
  const MoveStatistics &statistics = sampler.statistics(MoveKind::StitchMove);
  EXPECT_GT(statistics.topology_changes, 0U);
  EXPECT_GT(statistics.successor_changes, 2U * statistics.topology_changes);
}

TEST(InteractingSamplerTest, ExplicitCollectiveStitchValidatesStrands) {
  InteractingSampler sampler(sampler_model(0.0), 818);
  const std::array<ParticleId, 4> strands{0, 1, 2, 3};
  EXPECT_TRUE(sampler.k_stitch_update(4, strands, std::pair<double, double>{0.1, 0.7}));
  EXPECT_NO_THROW(sampler.state().validate(sampler.model().free));
  EXPECT_THROW(static_cast<void>(sampler.k_stitch_update(1)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sampler.k_stitch_update(7)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sampler.k_stitch_update(3, strands)), std::invalid_argument);
  const std::array<ParticleId, 3> duplicate{0, 1, 1};
  EXPECT_THROW(static_cast<void>(sampler.k_stitch_update(3, duplicate)), std::invalid_argument);
  EXPECT_THROW(sampler.stitch_sweep(1, 0.5, std::nullopt, 1, 0.05,
                                    StitchMixture{.strand_counts = {2, 2}, .strand_weights = {}}),
               std::invalid_argument);
  EXPECT_THROW(
      sampler.stitch_sweep(1, 0.5, std::nullopt, 1, 0.05,
                           StitchMixture{.strand_counts = {2, 3}, .strand_weights = {1.0}}),
      std::invalid_argument);
}

TEST(InteractingSamplerTest, RejectedGlobalMoveLeavesAcceptedStateUntouched) {
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

TEST(InteractingSamplerTest, RejectedLocalMovesLeaveStateCachesAndCountersUntouched) {
  const InteractingModel model{
      .free = {.particle_count = 5, .beta = 2.0, .linear_size = 2, .dimension = 1, .hopping = 1.0},
      .interaction = 100.0,
  };

  const auto expect_rejection = [&model](InteractingSampler &sampler, const MoveKind move,
                                         const auto &attempt) {
    bool observed_rejection = false;
    for (int trial = 0; trial < 400 && !observed_rejection; ++trial) {
      const ContinuousConfiguration before = sampler.state();
      const double overlap_before = sampler.pair_overlap();
      const double action_before = sampler.action();
      const MoveStatistics statistics_before = sampler.statistics(move);
      if (!attempt()) {
        observed_rejection = true;
        EXPECT_TRUE(equal_state(sampler.state(), before));
        EXPECT_DOUBLE_EQ(sampler.pair_overlap(), overlap_before);
        EXPECT_DOUBLE_EQ(sampler.action(), action_before);
        EXPECT_DOUBLE_EQ(pair_overlap_time(sampler.state(), model.free), overlap_before);
        EXPECT_TRUE(detail::InteractingSamplerTestAccess::occupancy_matches_state(sampler));

        const MoveStatistics &statistics_after = sampler.statistics(move);
        EXPECT_EQ(statistics_after.attempts, statistics_before.attempts + 1);
        EXPECT_EQ(statistics_after.accepts, statistics_before.accepts);
        EXPECT_EQ(statistics_after.topology_changes, statistics_before.topology_changes);
        EXPECT_EQ(statistics_after.successor_changes, statistics_before.successor_changes);
      }
    }
    EXPECT_TRUE(observed_rejection);
  };

  InteractingSampler segment_sampler(model, 1'927);
  expect_rejection(segment_sampler, MoveKind::SegmentMove, [&segment_sampler] {
    return segment_sampler.segment_update(0, std::pair<double, double>{0.2, 1.7});
  });

  InteractingSampler stitch_sampler(model, 2'927);
  expect_rejection(stitch_sampler, MoveKind::StitchMove, [&stitch_sampler] {
    return stitch_sampler.stitch_update(0, 1, std::pair<double, double>{0.2, 1.7});
  });
}

TEST(InteractingSamplerTest, ActionFailureAbandonsPreparedOccupancyReplacement) {
  const InteractingModel model{
      .free = {.particle_count = 2, .beta = 2.0, .linear_size = 1, .dimension = 1, .hopping = 0.0},
      .interaction = 0.0,
  };
  InteractingSampler sampler(model, 29);
  const ContinuousConfiguration before = sampler.state();
  const double overlap_before = sampler.pair_overlap();
  const double action_before = sampler.action();
  const MoveStatistics statistics_before = sampler.statistics(MoveKind::SegmentMove);
  detail::InteractingSamplerTestAccess::set_interaction(sampler,
                                                        std::numeric_limits<double>::max());

  EXPECT_THROW(static_cast<void>(sampler.whole_worldline_update(0)), std::overflow_error);
  detail::InteractingSamplerTestAccess::set_interaction(sampler, model.interaction);
  EXPECT_TRUE(equal_state(sampler.state(), before));
  EXPECT_DOUBLE_EQ(sampler.pair_overlap(), overlap_before);
  EXPECT_DOUBLE_EQ(sampler.action(), action_before);
  EXPECT_TRUE(detail::InteractingSamplerTestAccess::occupancy_matches_state(sampler));
  EXPECT_EQ(sampler.statistics(MoveKind::SegmentMove).attempts, statistics_before.attempts + 1);
  EXPECT_EQ(sampler.statistics(MoveKind::SegmentMove).accepts, statistics_before.accepts);
}

TEST(InteractingSamplerTest, TopologyPreparationFailureAbandonsPreparedReplacement) {
  InteractingSampler sampler(sampler_model(1.0), 39);
  const ContinuousConfiguration before = sampler.state();
  const double overlap_before = sampler.pair_overlap();
  const double action_before = sampler.action();
  const MoveStatistics statistics_before = sampler.statistics(MoveKind::StitchMove);

  EXPECT_THROW(static_cast<void>(
                   detail::InteractingSamplerTestAccess::try_invalid_topology_proposal(sampler)),
               std::invalid_argument);
  EXPECT_TRUE(equal_state(sampler.state(), before));
  EXPECT_DOUBLE_EQ(sampler.pair_overlap(), overlap_before);
  EXPECT_DOUBLE_EQ(sampler.action(), action_before);
  EXPECT_TRUE(detail::InteractingSamplerTestAccess::occupancy_matches_state(sampler));
  EXPECT_EQ(sampler.statistics(MoveKind::StitchMove).attempts, statistics_before.attempts);
  EXPECT_EQ(sampler.statistics(MoveKind::StitchMove).accepts, statistics_before.accepts);
}

TEST(InteractingSamplerTest, AcceptedCounterOverflowCannotPublishPreparedReplacement) {
  InteractingSampler sampler(sampler_model(0.0), 49);
  const ContinuousConfiguration before = sampler.state();
  const double overlap_before = sampler.pair_overlap();
  const double action_before = sampler.action();
  const MoveStatistics statistics_before = sampler.statistics(MoveKind::SegmentMove);
  detail::InteractingSamplerTestAccess::set_segment_accepts(
      sampler, std::numeric_limits<std::uint64_t>::max());

  EXPECT_THROW(static_cast<void>(sampler.whole_worldline_update(0)), std::overflow_error);
  EXPECT_TRUE(equal_state(sampler.state(), before));
  EXPECT_DOUBLE_EQ(sampler.pair_overlap(), overlap_before);
  EXPECT_DOUBLE_EQ(sampler.action(), action_before);
  EXPECT_TRUE(detail::InteractingSamplerTestAccess::occupancy_matches_state(sampler));
  EXPECT_EQ(sampler.statistics(MoveKind::SegmentMove).attempts, statistics_before.attempts + 1);
  EXPECT_EQ(sampler.statistics(MoveKind::SegmentMove).accepts,
            std::numeric_limits<std::uint64_t>::max());
}

TEST(InteractingSamplerTest, RunAppliesBurnInThinningAndReturnsTypedMeasurements) {
  InteractingSampler sampler(sampler_model(1.0), 77);
  const RunOptions options{
      .burn_in = 2,
      .thin = 3,
      .sweep = {.segment_updates = 2,
                .segment_fraction = 0.3,
                .cycle_updates = 1,
                .global_updates = 1,
                .stitch_mixture = {}},
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
  EXPECT_THROW(static_cast<void>(sampler.stitch_update(0, 0)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sampler.stitch_update(0, 1, std::nullopt, 0.0)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sampler.stitch_update(0, 1, std::pair<double, double>{0.8, 0.2})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sampler.time_shift_update(1.1)), std::invalid_argument);
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
    sampler.sweep(SweepOptions{.segment_updates = 3,
                               .segment_fraction = 0.5,
                               .cycle_updates = 1,
                               .global_updates = 1,
                               .stitch_mixture = {}});
  }
  EXPECT_EQ(sampler.state().event_count(), 0U);
  EXPECT_NO_THROW(sampler.state().validate(model.free));
  EXPECT_NEAR(sampler.action(), model.interaction * sampler.pair_overlap(), 1e-14);
}

} // namespace
} // namespace qmc
