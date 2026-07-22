#include "qmc/continuous_observables.hpp"
#include "qmc/interacting_sampler.hpp"
#include "qmc/path.hpp"

#include <array>
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

TEST(ContinuousDensityAccumulatorTest, MatchesOneParticleLehmannReference) {
  constexpr std::size_t sample_count = 40'000;
  const qmc::Model model(qmc::ModelParameters{
      .particle_count = 1,
      .beta = 0.8,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const qmc::CanonicalEnsemble ensemble(model);
  const qmc::MatsubaraModeSet modes(
      model.beta(), qmc::TorusLayout(model.linear_size(), model.dimension()),
      qmc::MatsubaraModeRequest{.momentum_indices = {{1}}, .frequency_indices = {0, 1}});
  const qmc::ContinuousMatsubaraPlan plan(modes);
  qmc::DensityMatsubaraAccumulator accumulator(model, modes);
  qmc::Random random(20260722);
  std::array<double, 2> observation_sums{};
  std::array<double, 2> observation_square_sums{};

  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const qmc::ContinuousConfiguration configuration =
        qmc::sample_ideal_continuous_configuration(ensemble, random);
    const qmc::ContinuousParticleModes values = qmc::continuous_particle_modes(configuration, plan);
    accumulator.observe(values);
    for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
      const double observation =
          std::norm(values.density(frequency, 0)) / (model.beta() * model.volume());
      observation_sums[frequency] += observation;
      observation_square_sums[frequency] += observation * observation;
    }
  }

  const qmc::ContinuousMatsubaraDensityCorrelations result = accumulator.finish();
  ASSERT_EQ(result.sample_count(), sample_count);
  const std::array exact{0.1915077492, 0.0217766472};
  const double count = static_cast<double>(sample_count);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    const double sample_mean = observation_sums[frequency] / count;
    const double sample_variance =
        (observation_square_sums[frequency] -
         (observation_sums[frequency] * observation_sums[frequency] / count)) /
        (count - 1.0);
    const double standard_error = std::sqrt(sample_variance / count);
    EXPECT_NEAR(result.at(frequency, 0), sample_mean, 1e-13);
    // Independent one-particle Lehmann values at q=2*pi/3, with a six-standard-
    // error bound from exact independent ideal samples.
    EXPECT_NEAR(result.at(frequency, 0), exact[frequency], 6.0 * standard_error);
    EXPECT_LT(standard_error, frequency == 0 ? 0.0010 : 0.00035);
  }
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

TEST(InteractingDistributionTest, StrongCouplingStitchKernelMatchesExactDiagonalization) {
  constexpr std::size_t burn_in = 5'000;
  constexpr std::size_t sample_count = 30'000;
  constexpr double beta = 2.0;
  constexpr double interaction = 16.0;
  const qmc::InteractingModel model{
      .free = qmc::Model(qmc::ModelParameters{
          .particle_count = 4,
          .beta = beta,
          .linear_size = 4,
          .dimension = 1,
          .hopping = 1.0,
      }),
      .interaction = interaction,
  };
  qmc::InteractingSampler sampler(model, 10101);
  const qmc::RandomSeamStitchOptions stitches{
      .updates = model.free.particle_count(),
      // Strong-coupling calibration: redraw a slab of duration 5/U.
      .fraction = 5.0 / (interaction * beta),
  };
  const qmc::SweepOptions local_geometry{
      .segment_updates = model.free.particle_count(),
      .segment_fraction = 0.35,
      .cycle_updates = 1,
      .global_updates = 0,
  };
  const auto advance = [&sampler, &stitches, &local_geometry] {
    sampler.random_seam_stitch_sweep(stitches);
    sampler.sweep(local_geometry);
  };

  for (std::size_t update = 0; update < burn_in; ++update) {
    advance();
  }

  double total_energy = 0.0;
  double kinetic_energy = 0.0;
  double interaction_energy = 0.0;
  double double_occupancy = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    advance();
    const auto value = sampler.observables();
    total_energy += value.total_energy;
    kinetic_energy += value.kinetic_energy;
    interaction_energy += value.interaction_energy;
    double_occupancy += value.double_occupancy_per_site;
  }

  const double denominator = static_cast<double>(sample_count);
  // Independent Fock-space diagonalization values from python/validate_interacting_ed.py.
  EXPECT_NEAR(total_energy / denominator, -1.0339175138, 0.075);
  EXPECT_NEAR(kinetic_energy / denominator, -2.1253604095, 0.14);
  EXPECT_NEAR(interaction_energy / denominator, 1.0914428957, 0.09);
  EXPECT_NEAR(double_occupancy / denominator, 0.0170537952, 0.0015);

  const qmc::MoveStatistics &stitch = sampler.statistics(qmc::MoveKind::StitchMove);
  const auto stitch_acceptance = stitch.acceptance();
  const auto topology_rate = stitch.topology_change_rate();
  ASSERT_TRUE(stitch_acceptance.has_value());
  ASSERT_TRUE(topology_rate.has_value());
  EXPECT_GT(*stitch_acceptance, 0.6);
  EXPECT_LT(*stitch_acceptance, 0.95);
  EXPECT_GT(*topology_rate, 0.03);
  EXPECT_EQ(sampler.statistics(qmc::MoveKind::GlobalMove).attempts, 0U);
}

} // namespace
