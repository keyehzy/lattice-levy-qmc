#include "qmc/interacting_sampler.hpp"
#include "qmc/path.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

namespace {

TEST(ContinuousBridgeDistributionTest, EventCountMatchesConditionedPoissonLaw) {
  constexpr double duration = 0.8;
  constexpr double hopping = 1.0;
  constexpr double lambda = duration * hopping;
  constexpr std::size_t sample_count = 30'000;

  double normalization = 0.0;
  double weighted_events = 0.0;
  double weight = 1.0;
  for (std::size_t pairs = 0; pairs < 100; ++pairs) {
    if (pairs > 0) {
      const auto count = static_cast<double>(pairs);
      weight *= (lambda * lambda) / (count * count);
    }
    normalization += weight;
    weighted_events += 2.0 * static_cast<double>(pairs) * weight;
  }
  const double exact_mean = weighted_events / normalization;

  qmc::Random random(8182);
  double event_sum = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const auto path = qmc::sample_continuous_bridge({0}, {0}, duration, hopping, random);
    event_sum += static_cast<double>(path.event_count());
  }
  EXPECT_NEAR(event_sum / static_cast<double>(sample_count), exact_mean, 0.035);
}

TEST(InteractingDistributionTest, MatchesSmallSystemExactDiagonalizationReference) {
  constexpr std::size_t burn_in = 2'000;
  constexpr std::size_t sample_count = 30'000;
  const qmc::InteractingModel model{
      .free = qmc::Model(qmc::ModelParameters{
          .particle_count = 2, .beta = 0.8, .linear_size = 3, .dimension = 1, .hopping = 1.0}),
      .interaction = 1.2,
  };
  qmc::InteractingSampler sampler(model, 20260717);
  for (std::size_t update = 0; update < burn_in; ++update) {
    static_cast<void>(sampler.global_update());
  }

  double total_energy = 0.0;
  double kinetic_energy = 0.0;
  double interaction_energy = 0.0;
  double pair_count = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    static_cast<void>(sampler.global_update());
    const auto value = sampler.observables();
    total_energy += value.total_energy;
    kinetic_energy += value.kinetic_energy;
    interaction_energy += value.interaction_energy;
    pair_count += value.pair_overlap_time / model.free.beta();
  }

  const double denominator = static_cast<double>(sample_count);
  // Independent Fock-space diagonalization values from python/validate_interacting_ed.py.
  EXPECT_NEAR(total_energy / denominator, -3.1421047841, 0.07);
  EXPECT_NEAR(kinetic_energy / denominator, -3.4840757439, 0.07);
  EXPECT_NEAR(interaction_energy / denominator, 0.3419709598, 0.018);
  EXPECT_NEAR(pair_count / denominator, 0.2849757998, 0.015);

  const auto acceptance = sampler.statistics(qmc::MoveKind::GlobalMove).acceptance();
  ASSERT_TRUE(acceptance.has_value());
  EXPECT_GT(*acceptance, 0.75);
  EXPECT_LT(*acceptance, 0.90);
}

} // namespace
