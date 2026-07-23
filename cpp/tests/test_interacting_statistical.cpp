#include "qmc/continuous_observables.hpp"
#include "qmc/interacting_sampler.hpp"
#include "qmc/path.hpp"

#include <algorithm>
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

TEST(ContinuousMatsubaraAccumulatorTest, MatchesOneParticleLehmannReferences) {
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
      qmc::MatsubaraModeRequest{.momentum_indices = {{0}, {1}}, .frequency_indices = {0, 1}});
  const qmc::ContinuousMatsubaraPlan plan(modes);
  qmc::DensityMatsubaraAccumulator density_accumulator(model, modes);
  qmc::HoppingResponseAccumulator hopping_accumulator(model, modes);
  qmc::Random random(20260722);
  std::array<double, 2> density_sums{};
  std::array<double, 2> density_square_sums{};
  std::array<double, 4> response_sums{};
  std::array<double, 4> response_square_sums{};
  double diamagnetic_sum = 0.0;
  double diamagnetic_square_sum = 0.0;

  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const qmc::ContinuousConfiguration configuration =
        qmc::sample_ideal_continuous_configuration(ensemble, random);
    const qmc::ContinuousParticleModes values = qmc::continuous_particle_modes(configuration, plan);
    density_accumulator.observe(values);
    hopping_accumulator.observe(values);
    for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
      const double density_observation =
          std::norm(values.density(frequency, 1)) / (model.beta() * model.volume());
      density_sums[frequency] += density_observation;
      density_square_sums[frequency] += density_observation * density_observation;
      for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
        const std::size_t mode = (frequency * modes.momentum_count()) + momentum;
        const double response_observation =
            std::norm(values.flux(frequency, momentum, 0)) / (model.beta() * model.volume());
        response_sums[mode] += response_observation;
        response_square_sums[mode] += response_observation * response_observation;
      }
    }
    const double diamagnetic_observation =
        static_cast<double>(values.axis_event_count(0)) / (model.beta() * model.volume());
    diamagnetic_sum += diamagnetic_observation;
    diamagnetic_square_sum += diamagnetic_observation * diamagnetic_observation;
  }

  const qmc::ContinuousMatsubaraDensityCorrelations density_result = density_accumulator.finish();
  const qmc::HoppingResponse hopping_result = hopping_accumulator.finish();
  ASSERT_EQ(density_result.sample_count(), sample_count);
  ASSERT_EQ(hopping_result.sample_count(), sample_count);
  const std::array exact_density{0.1915077492, 0.0217766472};
  const double count = static_cast<double>(sample_count);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    const double sample_mean = density_sums[frequency] / count;
    const double sample_variance = (density_square_sums[frequency] -
                                    (density_sums[frequency] * density_sums[frequency] / count)) /
                                   (count - 1.0);
    const double standard_error = std::sqrt(sample_variance / count);
    EXPECT_NEAR(density_result.at(frequency, 1), sample_mean, 1e-13);
    // Independent one-particle Lehmann values at q=2*pi/3, with a six-standard-
    // error bound from exact independent ideal samples.
    EXPECT_NEAR(density_result.at(frequency, 1), exact_density[frequency], 6.0 * standard_error);
    EXPECT_LT(standard_error, frequency == 0 ? 0.0010 : 0.00035);
  }

  // Independent one-particle source-derivative values. The mode order is
  // (n=0,q=0), (n=0,q=2*pi/3), (n=1,q=0), (n=1,q=2*pi/3).
  const std::array exact_response{0.3902364092, 0.0, 0.5130943014, 0.4477643599};
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      const std::size_t mode = (frequency * modes.momentum_count()) + momentum;
      const double sample_mean = response_sums[mode] / count;
      const double sample_variance =
          (response_square_sums[mode] - (response_sums[mode] * response_sums[mode] / count)) /
          (count - 1.0);
      const double standard_error = std::sqrt(std::max(0.0, sample_variance) / count);
      EXPECT_NEAR(hopping_result.flux_response(frequency, momentum, 0, 0).real(), sample_mean,
                  1e-13);
      EXPECT_EQ(hopping_result.flux_response(frequency, momentum, 0, 0).imag(), 0.0);
      if (exact_response[mode] == 0.0) {
        EXPECT_NEAR(hopping_result.flux_response(frequency, momentum, 0, 0).real(), 0.0, 1e-24);
      } else {
        EXPECT_NEAR(hopping_result.flux_response(frequency, momentum, 0, 0).real(),
                    exact_response[mode], 6.0 * standard_error);
        EXPECT_LT(standard_error, 0.01);
      }
    }
  }

  const double diamagnetic_mean = diamagnetic_sum / count;
  const double diamagnetic_variance =
      (diamagnetic_square_sum - (diamagnetic_sum * diamagnetic_sum / count)) / (count - 1.0);
  const double diamagnetic_standard_error = std::sqrt(diamagnetic_variance / count);
  EXPECT_NEAR(hopping_result.diamagnetic(0), diamagnetic_mean, 1e-13);
  EXPECT_NEAR(hopping_result.diamagnetic(0), 0.5130943014, 6.0 * diamagnetic_standard_error);
  EXPECT_LT(diamagnetic_standard_error, 0.005);
}

TEST(InteractingDistributionTest, MatchesSmallSystemExactDiagonalizationReference) {
  constexpr std::size_t burn_in = 2'000;
  constexpr std::size_t sample_count = 30'000;
  constexpr std::size_t block_size = 100;
  constexpr std::size_t block_count = sample_count / block_size;
  static_assert(sample_count % block_size == 0);
  const qmc::InteractingModel model{
      .free = qmc::Model(qmc::ModelParameters{
          .particle_count = 2, .beta = 0.8, .linear_size = 3, .dimension = 1, .hopping = 1.0}),
      .interaction = 1.2,
  };
  const qmc::MatsubaraModeSet modes(
      model.free.beta(), qmc::TorusLayout(model.free.linear_size(), model.free.dimension()),
      qmc::MatsubaraModeRequest{.momentum_indices = {{1}}, .frequency_indices = {0, 1}});
  const qmc::ContinuousMatsubaraPlan plan(modes);
  qmc::DensityMatsubaraAccumulator density_accumulator(model.free, modes);
  qmc::HoppingResponseAccumulator hopping_accumulator(model.free, modes);
  qmc::InteractingSampler sampler(model, 20260717);
  for (std::size_t update = 0; update < burn_in; ++update) {
    static_cast<void>(sampler.global_update());
  }

  double total_energy = 0.0;
  double kinetic_energy = 0.0;
  double interaction_energy = 0.0;
  double pair_count = 0.0;
  // Batch means account for autocorrelation in the Markov chain. Entries are
  // density (n=0,1), finite-frequency flux response, and the diamagnetic term.
  std::array<double, 4> block_sums{};
  std::array<double, 4> block_mean_sums{};
  std::array<double, 4> block_mean_square_sums{};
  const double normalization = model.free.beta() * static_cast<double>(model.free.volume());
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    static_cast<void>(sampler.global_update());
    const auto value = sampler.observables();
    total_energy += value.total_energy;
    kinetic_energy += value.kinetic_energy;
    interaction_energy += value.interaction_energy;
    pair_count += value.pair_overlap_time / model.free.beta();

    const qmc::ContinuousParticleModes modes_sample =
        qmc::continuous_particle_modes(sampler.state(), plan);
    density_accumulator.observe(modes_sample);
    hopping_accumulator.observe(modes_sample);
    const std::array observations{
        std::norm(modes_sample.density(0, 0)) / normalization,
        std::norm(modes_sample.density(1, 0)) / normalization,
        std::norm(modes_sample.flux(1, 0, 0)) / normalization,
        static_cast<double>(modes_sample.axis_event_count(0)) / normalization,
    };
    for (std::size_t observable = 0; observable < observations.size(); ++observable) {
      block_sums[observable] += observations[observable];
    }
    if ((sample + 1) % block_size == 0) {
      for (std::size_t observable = 0; observable < observations.size(); ++observable) {
        const double block_mean = block_sums[observable] / static_cast<double>(block_size);
        block_mean_sums[observable] += block_mean;
        block_mean_square_sums[observable] += block_mean * block_mean;
        block_sums[observable] = 0.0;
      }
    }
  }

  const double denominator = static_cast<double>(sample_count);
  // Independent Fock-space diagonalization values from python/validate_interacting_ed.py.
  EXPECT_NEAR(total_energy / denominator, -3.1421047841, 0.07);
  EXPECT_NEAR(kinetic_energy / denominator, -3.4840757439, 0.07);
  EXPECT_NEAR(interaction_energy / denominator, 0.3419709598, 0.018);
  EXPECT_NEAR(pair_count / denominator, 0.2849757998, 0.015);

  std::array<double, 4> block_standard_errors{};
  for (std::size_t observable = 0; observable < block_standard_errors.size(); ++observable) {
    const double block_sum = block_mean_sums[observable];
    const double centered_square_sum = block_mean_square_sums[observable] -
                                       (block_sum * block_sum / static_cast<double>(block_count));
    const double block_variance =
        std::max(0.0, centered_square_sum / static_cast<double>(block_count - 1));
    block_standard_errors[observable] =
        std::sqrt(block_variance / static_cast<double>(block_count));
  }

  const qmc::ContinuousMatsubaraDensityCorrelations density_result = density_accumulator.finish();
  const qmc::HoppingResponse hopping_result = hopping_accumulator.finish();
  ASSERT_EQ(density_result.sample_count(), sample_count);
  ASSERT_EQ(hopping_result.sample_count(), sample_count);
  const std::array exact_density{0.32798, 0.04773};
  for (std::size_t frequency = 0; frequency < exact_density.size(); ++frequency) {
    const double sample_mean = block_mean_sums[frequency] / static_cast<double>(block_count);
    EXPECT_NEAR(density_result.at(frequency, 0), sample_mean, 1e-13);
    EXPECT_NEAR(density_result.at(frequency, 0), exact_density[frequency],
                6.0 * block_standard_errors[frequency])
        << "blocked standard error = " << block_standard_errors[frequency];
  }
  const double response_mean = block_mean_sums[2] / static_cast<double>(block_count);
  EXPECT_NEAR(hopping_result.flux_response(1, 0, 0, 0).real(), response_mean, 1e-13);
  EXPECT_EQ(hopping_result.flux_response(1, 0, 0, 0).imag(), 0.0);
  EXPECT_NEAR(hopping_result.flux_response(1, 0, 0, 0).real(), 0.98151,
              6.0 * block_standard_errors[2])
      << "blocked standard error = " << block_standard_errors[2];

  const double diamagnetic_mean = block_mean_sums[3] / static_cast<double>(block_count);
  EXPECT_NEAR(hopping_result.diamagnetic(0), diamagnetic_mean, 1e-13);
  EXPECT_NEAR(hopping_result.diamagnetic(0), 1.16136, 6.0 * block_standard_errors[3])
      << "blocked standard error = " << block_standard_errors[3];

  // These caps keep the six-standard-error comparisons discriminating and
  // guard against a silently under-mixed chain.
  EXPECT_LT(block_standard_errors[0], 0.004);
  EXPECT_LT(block_standard_errors[1], 0.001);
  EXPECT_LT(block_standard_errors[2], 0.02);
  EXPECT_LT(block_standard_errors[3], 0.01);

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
