#include "qmc/configuration.hpp"
#include "qmc/free_boson.hpp"
#include "qmc/observables.hpp"
#include "qmc/random.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

struct BruteCanonicalResult {
  double partition = 0.0;
  double energy = 0.0;
  double energy_squared = 0.0;
  std::vector<double> occupations;
  std::vector<double> occupation_squared;
};

BruteCanonicalResult enumerate_fock_states(const qmc::Model &model) {
  const auto volume = model.volume();
  std::vector<double> energies(volume);
  for (std::size_t momentum = 0; momentum < volume; ++momentum) {
    const double angle = 2.0 * std::numbers::pi * static_cast<double>(momentum) /
                         static_cast<double>(model.linear_size);
    energies[momentum] = -2.0 * model.hopping * std::cos(angle);
  }
  std::vector<std::size_t> occupation(volume);
  BruteCanonicalResult result{
      .occupations = std::vector<double>(volume),
      .occupation_squared = std::vector<double>(volume),
  };
  std::function<void(std::size_t, std::size_t)> enumerate = [&](const std::size_t mode,
                                                                const std::size_t remaining) {
    if (mode + 1 == volume) {
      occupation[mode] = remaining;
      double energy = 0.0;
      for (std::size_t index = 0; index < volume; ++index) {
        energy += static_cast<double>(occupation[index]) * energies[index];
      }
      const double weight = std::exp(-model.beta * energy);
      result.partition += weight;
      result.energy += weight * energy;
      result.energy_squared += weight * energy * energy;
      for (std::size_t index = 0; index < volume; ++index) {
        const auto value = static_cast<double>(occupation[index]);
        result.occupations[index] += weight * value;
        result.occupation_squared[index] += weight * value * value;
      }
      return;
    }
    for (std::size_t value = 0; value <= remaining; ++value) {
      occupation[mode] = value;
      enumerate(mode + 1, remaining - value);
    }
  };
  enumerate(0, model.particle_count);

  result.energy /= result.partition;
  result.energy_squared /= result.partition;
  for (std::size_t index = 0; index < volume; ++index) {
    result.occupations[index] /= result.partition;
    result.occupation_squared[index] /= result.partition;
  }
  return result;
}

TEST(CanonicalObservablesTest, MatchesFockSpaceEnumeration) {
  const qmc::Model model{
      .particle_count = 3,
      .beta = 0.7,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 0.8,
  };
  const qmc::CanonicalEnsemble canonical(model);
  const auto thermodynamics = qmc::canonical_thermodynamics(canonical);
  const auto momentum = qmc::momentum_distribution(canonical);
  const auto brute = enumerate_fock_states(model);
  const auto particles = model.particle_count;

  EXPECT_NEAR(std::exp(canonical.log_partition(particles)), brute.partition, 2e-13);
  EXPECT_NEAR(thermodynamics.free_energy[particles], -std::log(brute.partition) / model.beta,
              2e-13);
  EXPECT_NEAR(thermodynamics.energy[particles], brute.energy, 2e-13);
  EXPECT_NEAR(thermodynamics.heat_capacity[particles],
              model.beta * model.beta * (brute.energy_squared - (brute.energy * brute.energy)),
              3e-13);
  EXPECT_NEAR(thermodynamics.entropy[particles],
              std::log(brute.partition) + (model.beta * brute.energy), 3e-13);
  EXPECT_NEAR(thermodynamics.addition_chemical_potential[particles],
              thermodynamics.free_energy[particles] - thermodynamics.free_energy[particles - 1],
              1e-15);

  ASSERT_EQ(momentum.modes.size(), brute.occupations.size());
  for (std::size_t mode = 0; mode < momentum.modes.size(); ++mode) {
    EXPECT_NEAR(momentum.modes[mode].occupation, brute.occupations[mode], 3e-13);
    EXPECT_NEAR(momentum.modes[mode].occupation_variance,
                brute.occupation_squared[mode] -
                    (brute.occupations[mode] * brute.occupations[mode]),
                5e-13);
  }
  EXPECT_NEAR(momentum.kinetic_energy, brute.energy, 3e-13);
}

TEST(CanonicalObservablesTest, SatisfiesMomentumDensityMatrixAndCycleNormalizations) {
  const qmc::Model model{
      .particle_count = 7,
      .beta = 1.1,
      .linear_size = 5,
      .dimension = 2,
      .hopping = 0.9,
  };
  const qmc::CanonicalEnsemble canonical(model);
  const auto momentum = qmc::momentum_distribution(canonical);
  const auto density_matrix = qmc::one_body_density_matrix(canonical);
  const auto cycles = qmc::exact_cycle_statistics(canonical);

  const double occupation_sum = std::accumulate(
      momentum.modes.begin(), momentum.modes.end(), 0.0,
      [](const double sum, const qmc::MomentumMode &mode) { return sum + mode.occupation; });
  EXPECT_NEAR(occupation_sum, static_cast<double>(model.particle_count), 2e-12);
  EXPECT_NEAR(momentum.condensate_density,
              momentum.condensate_occupation / static_cast<double>(model.volume()), 1e-15);
  EXPECT_GT(momentum.coherence_length, 0.0);
  ASSERT_EQ(density_matrix.size(), model.volume());
  EXPECT_NEAR(density_matrix.front().value,
              static_cast<double>(model.particle_count) / static_cast<double>(model.volume()),
              2e-13);
  const double density_matrix_sum = std::accumulate(
      density_matrix.begin(), density_matrix.end(), 0.0,
      [](const double sum, const qmc::OneBodyDensityPoint &point) { return sum + point.value; });
  EXPECT_NEAR(density_matrix_sum, momentum.condensate_occupation, 3e-12);

  const double expected_particles =
      std::accumulate(cycles.expected_particles.begin(), cycles.expected_particles.end(), 0.0);
  const double probability_sum =
      std::accumulate(cycles.particle_probability.begin(), cycles.particle_probability.end(), 0.0);
  EXPECT_NEAR(expected_particles, static_cast<double>(model.particle_count), 2e-13);
  EXPECT_NEAR(probability_sum, 1.0, 2e-13);
}

TEST(CanonicalObservablesTest, TwistCurvatureMatchesFiniteDifference) {
  const qmc::Model model{
      .particle_count = 4,
      .beta = 0.9,
      .linear_size = 5,
      .dimension = 2,
      .hopping = 0.7,
  };
  const qmc::CanonicalEnsemble canonical(model);
  const double curvature = qmc::twist_free_energy_curvature(canonical, 0);
  constexpr double step = 1e-3;
  const std::vector<double> zero(model.dimension);
  std::vector<double> positive(model.dimension);
  std::vector<double> negative(model.dimension);
  positive[0] = step;
  negative[0] = -step;
  const double free_zero = -qmc::log_canonical_partition_twisted(canonical, zero) / model.beta;
  const double free_positive =
      -qmc::log_canonical_partition_twisted(canonical, positive) / model.beta;
  const double free_negative =
      -qmc::log_canonical_partition_twisted(canonical, negative) / model.beta;
  const double finite_difference =
      (free_positive + free_negative - (2.0 * free_zero)) / (step * step);
  EXPECT_GT(curvature, 0.0);
  EXPECT_NEAR(curvature, finite_difference, 2e-8);
}

TEST(ConfigurationObservablesTest, ExactSampleInvariantsHoldOnRetainedGrid) {
  const qmc::Model model{
      .particle_count = 6,
      .beta = 1.2,
      .linear_size = 4,
      .dimension = 2,
      .hopping = 0.9,
  };
  qmc::Random random(1807);
  const auto configuration = qmc::sample_ideal_boson_configuration(model, 7, random);
  const auto histogram = qmc::sampled_cycle_histogram(configuration);
  const auto winding = qmc::total_winding(configuration);
  const auto equal_time = qmc::equal_time_observables(configuration);
  const auto imaginary_time = qmc::retained_density_correlations(configuration);
  const auto matsubara = qmc::retained_grid_matsubara_transform(model, imaginary_time);
  const auto retained_geometry = qmc::retained_geometry_observables(configuration);
  const auto cycle_geometry = qmc::retained_cycle_geometry(configuration);

  std::size_t particles_in_cycles = 0;
  for (std::size_t length = 1; length < histogram.size(); ++length) {
    particles_in_cycles += length * histogram[length];
  }
  EXPECT_EQ(particles_in_cycles, model.particle_count);
  EXPECT_GE(qmc::longest_cycle_length(configuration), 1U);
  qmc::Site winding_reference(model.dimension);
  for (std::size_t cycle = 0; cycle < configuration.topology().cycles().size(); ++cycle) {
    const qmc::Site cycle_winding = configuration.cycle_winding(cycle);
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      winding_reference[axis] += cycle_winding[axis];
    }
  }
  EXPECT_EQ(winding, winding_reference);

  EXPECT_NEAR(std::accumulate(equal_time.site_density.begin(), equal_time.site_density.end(), 0.0),
              static_cast<double>(model.particle_count), 2e-13);
  EXPECT_NEAR(std::accumulate(equal_time.onsite_occupation_probability.begin(),
                              equal_time.onsite_occupation_probability.end(), 0.0),
              1.0, 2e-13);
  EXPECT_NEAR(equal_time.static_structure_factor.front(), static_cast<double>(model.particle_count),
              2e-13);
  const double pair_sum =
      std::accumulate(equal_time.pair_correlation.begin(), equal_time.pair_correlation.end(), 0.0);
  EXPECT_NEAR(pair_sum,
              static_cast<double>(model.volume() * (model.particle_count - 1)) /
                  static_cast<double>(model.particle_count),
              2e-12);

  ASSERT_EQ(imaginary_time.time_points, configuration.time_links_per_beta());
  ASSERT_EQ(imaginary_time.spatial_points, model.volume());
  const double density =
      static_cast<double>(model.particle_count) / static_cast<double>(model.volume());
  for (std::size_t displacement = 0; displacement < model.volume(); ++displacement) {
    const double expected =
        (density * density * (equal_time.pair_correlation[displacement] - 1.0)) +
        (displacement == 0 ? density : 0.0);
    EXPECT_NEAR(imaginary_time.connected_density[displacement], expected, 2e-13);
  }
  for (std::size_t lag = 0; lag < imaginary_time.time_points; ++lag) {
    const auto begin = imaginary_time.connected_density.begin() +
                       static_cast<std::ptrdiff_t>(lag * model.volume());
    EXPECT_NEAR(std::accumulate(begin, begin + static_cast<std::ptrdiff_t>(model.volume()), 0.0),
                0.0, 2e-13);
  }
  for (std::size_t frequency = 0; frequency < matsubara.frequencies.size(); ++frequency) {
    EXPECT_NEAR(std::abs(matsubara.values[frequency * model.volume()]), 0.0, 3e-12);
  }

  ASSERT_EQ(retained_geometry.time_points, configuration.time_links_per_beta());
  EXPECT_DOUBLE_EQ(retained_geometry.mean_square_displacement.front(), 0.0);
  EXPECT_DOUBLE_EQ(retained_geometry.return_probability.front(), 1.0);
  for (std::size_t time = 0; time < retained_geometry.time_points; ++time) {
    const auto begin = retained_geometry.displacement_probability.begin() +
                       static_cast<std::ptrdiff_t>(time * model.volume());
    EXPECT_NEAR(std::accumulate(begin, begin + static_cast<std::ptrdiff_t>(model.volume()), 0.0),
                1.0, 2e-13);
    EXPECT_NEAR(retained_geometry.return_probability[time], *begin, 2e-13);
    EXPECT_GE(retained_geometry.mean_square_displacement[time], 0.0);
  }
  ASSERT_EQ(cycle_geometry.size(), configuration.topology().cycles().size());
  for (std::size_t cycle = 0; cycle < cycle_geometry.size(); ++cycle) {
    EXPECT_EQ(cycle_geometry[cycle].length, configuration.topology().cycles()[cycle].size());
    EXPECT_EQ(cycle_geometry[cycle].winding, configuration.cycle_winding(cycle));
    EXPECT_GE(cycle_geometry[cycle].radius_of_gyration_squared, 0.0);
    EXPECT_GE(cycle_geometry[cycle].maximum_radius_squared,
              cycle_geometry[cycle].radius_of_gyration_squared);
  }
}

TEST(ConfigurationObservablesTest, RejectsWrappedMatsubaraGridExtent) {
  const qmc::Model model{
      .particle_count = 0,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 0.0,
  };
  const qmc::ImaginaryTimeDensityCorrelations correlations{
      .time_points = (std::numeric_limits<std::size_t>::max() / model.volume()) + 1,
      .spatial_points = model.volume(),
      .connected_density = {},
  };

  EXPECT_THROW(static_cast<void>(qmc::retained_grid_matsubara_transform(model, correlations)),
               std::overflow_error);
}

TEST(CanonicalObservablesTest, HandlesEmptySystemAndRejectsUndefinedTemperatureQuantities) {
  const qmc::Model zero_temperature_parameter{
      .particle_count = 2,
      .beta = 0.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  };
  const qmc::CanonicalEnsemble beta_zero(zero_temperature_parameter);
  EXPECT_THROW(static_cast<void>(qmc::canonical_thermodynamics(beta_zero)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::twist_free_energy_curvature(beta_zero, 0)),
               std::invalid_argument);

  const qmc::Model empty{
      .particle_count = 0,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  };
  const qmc::CanonicalEnsemble empty_ensemble(empty);
  const auto thermodynamics = qmc::canonical_thermodynamics(empty_ensemble);
  const auto momentum = qmc::momentum_distribution(empty_ensemble);
  const auto density_matrix = qmc::one_body_density_matrix(empty_ensemble);
  EXPECT_DOUBLE_EQ(thermodynamics.free_energy.front(), 0.0);
  EXPECT_DOUBLE_EQ(momentum.condensate_occupation, 0.0);
  EXPECT_TRUE(std::ranges::all_of(
      momentum.modes, [](const qmc::MomentumMode &mode) { return mode.occupation == 0.0; }));
  EXPECT_TRUE(std::ranges::all_of(
      density_matrix, [](const qmc::OneBodyDensityPoint &point) { return point.value == 0.0; }));
}

} // namespace
