#include "canonical_recursion.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace qmc::detail {
namespace {

TEST(CanonicalRecursionTest, PreservesTheEmptySystemBaseCase) {
  const std::array log_cycle_weights{-std::numeric_limits<double>::infinity()};

  const auto log_partitions = canonical_log_partitions(log_cycle_weights);

  ASSERT_EQ(log_partitions.size(), 1U);
  EXPECT_DOUBLE_EQ(log_partitions[0], 0.0);
}

TEST(CanonicalRecursionTest, MatchesAClosedFormAcrossEveryPrefix) {
  constexpr std::size_t particle_count = 5;
  constexpr double log_scale = 700.0;
  std::vector<double> log_cycle_weights(particle_count + 1,
                                        -std::numeric_limits<double>::infinity());
  for (std::size_t length = 1; length <= particle_count; ++length) {
    log_cycle_weights[length] = static_cast<double>(length) * log_scale;
  }

  const auto log_partitions = canonical_log_partitions(log_cycle_weights);

  ASSERT_EQ(log_partitions.size(), particle_count + 1);
  for (std::size_t particles = 0; particles <= particle_count; ++particles) {
    EXPECT_NEAR(log_partitions[particles], static_cast<double>(particles) * log_scale, 1e-12);
  }
}

TEST(CanonicalRecursionTest, AllowsZeroWeightForIndividualCycleLengths) {
  const double negative_infinity = -std::numeric_limits<double>::infinity();
  const std::array log_cycle_weights{negative_infinity, 0.0, negative_infinity};

  const auto log_partitions = canonical_log_partitions(log_cycle_weights);

  ASSERT_EQ(log_partitions.size(), 3U);
  EXPECT_DOUBLE_EQ(log_partitions[1], 0.0);
  EXPECT_DOUBLE_EQ(log_partitions[2], -std::log(2.0));
}

TEST(CanonicalRecursionTest, RejectsMissingOrNonFiniteNumericalInput) {
  EXPECT_THROW(static_cast<void>(canonical_log_partitions(std::span<const double>{})),
               std::invalid_argument);

  const double negative_infinity = -std::numeric_limits<double>::infinity();
  const std::array positive_infinity{negative_infinity, std::numeric_limits<double>::infinity()};
  const std::array not_a_number{negative_infinity, std::numeric_limits<double>::quiet_NaN()};
  const std::array zero_mass{negative_infinity, negative_infinity};
  EXPECT_THROW(static_cast<void>(canonical_log_partitions(positive_infinity)), std::overflow_error);
  EXPECT_THROW(static_cast<void>(canonical_log_partitions(not_a_number)), std::overflow_error);
  EXPECT_THROW(static_cast<void>(canonical_log_partitions(zero_mass)), std::overflow_error);
}

} // namespace
} // namespace qmc::detail
