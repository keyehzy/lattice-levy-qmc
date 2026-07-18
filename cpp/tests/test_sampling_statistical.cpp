#include "qmc/free_boson.hpp"
#include "qmc/free_numerics.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

namespace {

TEST(MidpointDistributionTest, MatchesExactBesselPmf) {
  constexpr qmc::Coord minimum = -8;
  constexpr qmc::Coord maximum = 10;
  constexpr std::size_t sample_count = 100'000;
  std::vector<qmc::Coord> coordinates;
  for (qmc::Coord coordinate = minimum; coordinate <= maximum; ++coordinate) {
    coordinates.push_back(coordinate);
  }
  const auto exact = qmc::exact_midpoint_pmf_window(0, 2, 0.5, 0.5, 1.0, coordinates);
  ASSERT_GT(std::accumulate(exact.begin(), exact.end(), 0.0), 1.0 - 1e-13);

  std::vector<std::size_t> counts(coordinates.size(), 0);
  qmc::Random random(12345);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const qmc::Coord midpoint = qmc::sample_midpoint_covering_1d(0, 2, 0.5, 0.5, 1.0, random);
    if (midpoint >= minimum && midpoint <= maximum) {
      ++counts[static_cast<std::size_t>(midpoint - minimum)];
    }
  }
  for (std::size_t index = 0; index < exact.size(); ++index) {
    const double empirical = static_cast<double>(counts[index]) / sample_count;
    EXPECT_NEAR(empirical, exact[index], 0.006);
  }
}

TEST(MidpointDistributionTest, HandlesWidelySeparatedEndpoints) {
  constexpr std::size_t sample_count = 20'000;
  qmc::Random random(9);
  double sum = 0.0;
  double square_sum = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const double midpoint =
        static_cast<double>(qmc::sample_midpoint_covering_1d(0, 100, 0.5, 0.5, 10.0, random));
    sum += midpoint;
    square_sum += midpoint * midpoint;
  }
  const double mean = sum / sample_count;
  const double variance = (square_sum / sample_count) - (mean * mean);
  EXPECT_NEAR(mean, 50.0, 0.25);
  EXPECT_GT(std::sqrt(variance), 3.0);
}

TEST(WindingDistributionTest, MatchesScaledBesselWeights) {
  constexpr qmc::Coord linear_size = 3;
  constexpr double duration = 1.7;
  constexpr double hopping = 0.8;
  constexpr qmc::Coord support = 8;
  constexpr std::size_t sample_count = 60'000;
  const double argument = 2.0 * hopping * duration;

  std::vector<double> exact(static_cast<std::size_t>((2 * support) + 1));
  for (qmc::Coord winding = -support; winding <= support; ++winding) {
    exact[static_cast<std::size_t>(winding + support)] = qmc::scaled_modified_bessel_i(
        static_cast<std::uint64_t>(std::abs(winding) * linear_size), argument);
  }
  const double normalization = std::accumulate(exact.begin(), exact.end(), 0.0);
  for (double &probability : exact) {
    probability /= normalization;
  }

  std::vector<std::size_t> counts(exact.size(), 0);
  qmc::Random random(3847);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const qmc::Coord winding = qmc::sample_winding_1d(linear_size, duration, hopping, random);
    ASSERT_GE(winding, -support);
    ASSERT_LE(winding, support);
    ++counts[static_cast<std::size_t>(winding + support)];
  }
  for (std::size_t index = 0; index < exact.size(); ++index) {
    const double empirical = static_cast<double>(counts[index]) / sample_count;
    EXPECT_NEAR(empirical, exact[index], 0.006);
  }
}

TEST(CycleDistributionTest, MatchesCanonicalLengthProbabilities) {
  constexpr std::size_t sample_count = 60'000;
  const qmc::Model model{
      .particle_count = 5,
      .beta = 0.8,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 0.9,
  };
  const auto table = qmc::canonical_table(model);
  std::vector<double> exact(model.particle_count);
  for (std::size_t length = 1; length <= model.particle_count; ++length) {
    exact[length - 1] = std::exp(table.log_z[length] + table.log_Z[model.particle_count - length] -
                                 std::log(static_cast<double>(model.particle_count)) -
                                 table.log_Z[model.particle_count]);
  }

  std::vector<std::size_t> counts(model.particle_count, 0);
  qmc::Random random(774);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const auto cycles = qmc::sample_cycle_labels(model.particle_count, table, random);
    ASSERT_FALSE(cycles.empty());
    ++counts[cycles.front().size() - 1];
  }
  for (std::size_t index = 0; index < exact.size(); ++index) {
    const double empirical = static_cast<double>(counts[index]) / sample_count;
    EXPECT_NEAR(empirical, exact[index], 0.007);
  }
}

TEST(TorusMidpointDistributionTest, MatchesFiniteRingKernel) {
  constexpr qmc::Coord linear_size = 5;
  constexpr std::size_t sample_count = 50'000;
  std::vector<double> exact(static_cast<std::size_t>(linear_size));
  for (qmc::Coord site = 0; site < linear_size; ++site) {
    exact[static_cast<std::size_t>(site)] =
        qmc::periodic_kernel_scaled_1d(site, 0.4, linear_size, 0.7) *
        qmc::periodic_kernel_scaled_1d(2 - site, 0.9, linear_size, 0.7);
  }
  const double normalization = std::accumulate(exact.begin(), exact.end(), 0.0);
  for (double &probability : exact) {
    probability /= normalization;
  }

  std::vector<std::size_t> counts(exact.size(), 0);
  qmc::Random random(8009);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const auto midpoint = qmc::sample_midpoint_torus_1d(0, 2, 0.4, 0.9, linear_size, 0.7, random);
    ++counts[static_cast<std::size_t>(midpoint)];
  }
  for (std::size_t index = 0; index < exact.size(); ++index) {
    const double empirical = static_cast<double>(counts[index]) / sample_count;
    EXPECT_NEAR(empirical, exact[index], 0.007);
  }
}

} // namespace
