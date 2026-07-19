#include "qmc/free_boson.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

static_assert(!std::is_aggregate_v<qmc::CanonicalEnsemble>);
static_assert(std::is_same_v<decltype(std::declval<const qmc::CanonicalEnsemble &>().model()),
                             const qmc::Model &>);

std::vector<std::size_t> permutation_cycle_lengths(const std::vector<qmc::ParticleId> &values) {
  std::vector<bool> seen(values.size(), false);
  std::vector<std::size_t> lengths;
  for (std::size_t start = 0; start < values.size(); ++start) {
    if (seen[start]) {
      continue;
    }
    std::size_t current = start;
    std::size_t length = 0;
    while (!seen[current]) {
      seen[current] = true;
      ++length;
      current = values[current];
    }
    lengths.push_back(length);
  }
  return lengths;
}

bool cycles_partition_labels(const std::vector<qmc::Cycle> &cycles,
                             const std::size_t particle_count) {
  std::vector<std::size_t> occurrences(particle_count, 0);
  for (const auto &cycle : cycles) {
    if (cycle.empty()) {
      return false;
    }
    for (const auto label : cycle) {
      if (static_cast<std::size_t>(label) >= particle_count) {
        return false;
      }
      ++occurrences[label];
    }
  }
  return std::ranges::all_of(occurrences, [](const std::size_t count) { return count == 1; });
}

TEST(FreeBosonTest, TorusTraceMatchesPythonAndDirectMomentumSum) {
  constexpr qmc::Coord linear_size = 5;
  constexpr std::size_t dimension = 2;
  constexpr double hopping = 0.7;
  constexpr double duration = 1.3;
  const double got = qmc::log_one_particle_trace(duration, linear_size, dimension, hopping);
  EXPECT_NEAR(got, 4.633054647408159, 2e-14);

  double direct_1d = 0.0;
  for (qmc::Coord momentum = 0; momentum < linear_size; ++momentum) {
    const double angle =
        2.0 * std::numbers::pi * static_cast<double>(momentum) / static_cast<double>(linear_size);
    direct_1d += std::exp(2.0 * hopping * duration * std::cos(angle));
  }
  EXPECT_NEAR(std::exp(got), direct_1d * direct_1d, 2e-13);
}

TEST(FreeBosonTest, CanonicalEnsembleMatchesPythonReference) {
  const qmc::Model model{
      .particle_count = 5,
      .beta = 0.8,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 0.9,
  };
  const qmc::CanonicalEnsemble ensemble(model);
  const auto log_z = ensemble.log_cycle_weights();
  const auto log_Z = ensemble.log_partitions();
  const std::array log_z_reference{
      -std::numeric_limits<double>::infinity(),
      1.8652613825726467,
      2.9892315869240047,
      4.3464244330862565,
      5.766292314502728,
      7.201492614503655,
  };
  const std::array log_Z_reference{0.0,
                                   1.8652613825726467,
                                   3.4270488377795676,
                                   4.903593650305713,
                                   6.354339970092079,
                                   7.797407636365048};
  ASSERT_EQ(log_z.size(), log_z_reference.size());
  ASSERT_EQ(log_Z.size(), log_Z_reference.size());
  for (std::size_t index = 1; index < log_z_reference.size(); ++index) {
    EXPECT_NEAR(log_z[index], log_z_reference[index], 3e-14);
  }
  for (std::size_t index = 0; index < log_Z_reference.size(); ++index) {
    EXPECT_NEAR(log_Z[index], log_Z_reference[index], 3e-14);
  }
}

TEST(FreeBosonTest, CanonicalEnsembleOwnsItsValidatedModel) {
  qmc::Model model{
      .particle_count = 3,
      .beta = 0.8,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 0.9,
  };
  const qmc::CanonicalEnsemble ensemble(model);
  model.beta = 1.7;
  model.hopping = 0.2;

  EXPECT_DOUBLE_EQ(ensemble.model().beta, 0.8);
  EXPECT_DOUBLE_EQ(ensemble.model().hopping, 0.9);
  EXPECT_TRUE(std::isfinite(ensemble.log_partition(3)));
}

TEST(FreeBosonTest, CanonicalRecursionMatchesPermutationEnumeration) {
  constexpr std::size_t particle_count = 5;
  const qmc::Model model{
      .particle_count = particle_count,
      .beta = 0.8,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 0.9,
  };
  const qmc::CanonicalEnsemble ensemble(model);
  const auto log_z = ensemble.log_cycle_weights();
  std::vector<qmc::ParticleId> permutation{0, 1, 2, 3, 4};
  double total = 0.0;
  do {
    double weight = 1.0;
    for (const auto length : permutation_cycle_lengths(permutation)) {
      weight *= std::exp(log_z[length]);
    }
    total += weight;
  } while (std::ranges::next_permutation(permutation).found);
  total /= 120.0;
  EXPECT_NEAR(std::exp(ensemble.log_partition(particle_count)), total, 2e-10);
}

TEST(FreeBosonTest, SampledCyclesPartitionLabels) {
  const qmc::Model model{
      .particle_count = 19,
      .beta = 0.6,
      .linear_size = 7,
      .dimension = 2,
      .hopping = 1.1,
  };
  const qmc::CanonicalEnsemble ensemble(model);
  qmc::Random random(2026);
  const auto cycles = ensemble.sample_cycles(random);

  EXPECT_TRUE(cycles_partition_labels(cycles, model.particle_count));
}

TEST(FreeBosonTest, CanonicalPrefixReuseIsExplicitAndBounded) {
  const qmc::Model model{
      .particle_count = 7,
      .beta = 0.6,
      .linear_size = 5,
      .dimension = 2,
      .hopping = 0.9,
  };
  const qmc::CanonicalEnsemble ensemble(model);
  qmc::Model prefix_model = model;
  prefix_model.particle_count = 3;
  const qmc::CanonicalEnsemble prefix(prefix_model);
  qmc::Random reused_random(27);
  qmc::Random direct_random(27);
  const auto reused_cycles = ensemble.sample_cycles(3, reused_random);
  const auto direct_cycles = prefix.sample_cycles(direct_random);

  EXPECT_TRUE(cycles_partition_labels(reused_cycles, 3));
  EXPECT_EQ(reused_cycles, direct_cycles);
  EXPECT_DOUBLE_EQ(ensemble.log_partition(3), prefix.log_partition(3));
  EXPECT_THROW(static_cast<void>(ensemble.sample_cycles(8, reused_random)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(ensemble.log_partition(8)), std::out_of_range);
}

TEST(FreeBosonTest, CanonicalConstructionRejectsNonFiniteRecursionAtItsSource) {
  const qmc::Model model{
      .particle_count = 2,
      .beta = std::numeric_limits<double>::max(),
      .linear_size = 2,
      .dimension = 1,
      .hopping = 0.0,
  };

  EXPECT_THROW(static_cast<void>(qmc::CanonicalEnsemble(model)), std::overflow_error);
}

TEST(FreeBosonTest, EmptyCanonicalSystemIsValid) {
  const qmc::Model model{
      .particle_count = 0,
      .beta = 0.0,
      .linear_size = 3,
      .dimension = 2,
      .hopping = 0.0,
  };
  const qmc::CanonicalEnsemble ensemble(model);
  EXPECT_EQ(ensemble.log_cycle_weights().size(), 1U);
  ASSERT_EQ(ensemble.log_partitions().size(), 1U);
  EXPECT_DOUBLE_EQ(ensemble.log_partition(0), 0.0);
  qmc::Random random(4);
  EXPECT_TRUE(ensemble.sample_cycles(random).empty());
}

TEST(FreeBosonTest, WindingHandlesZeroWeightAndLimits) {
  qmc::Random random(9);
  EXPECT_EQ(qmc::sample_winding_1d(5, 0.0, 1.0, random), 0);
  EXPECT_EQ(qmc::sample_winding_1d(5, 3.0, 0.0, random), 0);
  const qmc::NumericalOptions options{
      .tail_tolerance = 1e-14,
      .max_bessel_terms = 100,
      .max_winding = 1,
  };
  EXPECT_THROW(static_cast<void>(qmc::sample_winding_1d(1, 10.0, 1.0, random, options)),
               std::runtime_error);
}

TEST(FreeBosonTest, WindingWeightsMatchPythonReference) {
  constexpr qmc::Coord linear_size = 3;
  constexpr double argument = 2.0 * 0.8 * 1.7;
  constexpr qmc::Coord support = 8;
  std::vector<double> probabilities(static_cast<std::size_t>((2 * support) + 1));
  for (qmc::Coord winding = -support; winding <= support; ++winding) {
    probabilities[static_cast<std::size_t>(winding + support)] = qmc::scaled_modified_bessel_i(
        static_cast<std::uint64_t>(std::abs(winding) * linear_size), argument);
  }
  const double normalization = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
  for (double &probability : probabilities) {
    probability /= normalization;
  }

  EXPECT_NEAR(probabilities[6], 2.1788482402850134e-03, 2e-15);
  EXPECT_NEAR(probabilities[7], 1.2480769643018139e-01, 2e-14);
  EXPECT_NEAR(probabilities[8], 7.4600672822461245e-01, 3e-14);
  EXPECT_NEAR(probabilities[9], 1.2480769643018139e-01, 2e-14);
  EXPECT_NEAR(probabilities[10], 2.1788482402850134e-03, 2e-15);
}

} // namespace
