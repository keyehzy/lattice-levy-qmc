#include "qmc/random.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <stdexcept>

namespace {

TEST(RandomTest, DiscreteLogIndexRejectsInvalidWeightsBeforeDrawing) {
  const double negative_infinity = -std::numeric_limits<double>::infinity();
  const std::array nan_weight{0.0, std::numeric_limits<double>::quiet_NaN()};
  const std::array positive_infinity{0.0, std::numeric_limits<double>::infinity()};
  const std::array no_probability_mass{negative_infinity, negative_infinity};
  qmc::Random random(9182);
  qmc::Random control(9182);

  EXPECT_THROW(static_cast<void>(random.discrete_log_index(std::span<const double>{})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(random.discrete_log_index(nan_weight)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(random.discrete_log_index(positive_infinity)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(random.discrete_log_index(no_probability_mass)),
               std::invalid_argument);
  EXPECT_DOUBLE_EQ(random.uniform_unit(), control.uniform_unit());
}

TEST(RandomTest, DiscreteLogIndexHandlesZeroMassAndExtremeCommonOffsets) {
  const double negative_infinity = -std::numeric_limits<double>::infinity();
  const std::array deterministic{negative_infinity, -10'000.0, negative_infinity};
  qmc::Random deterministic_random(17);
  for (int draw = 0; draw < 20; ++draw) {
    EXPECT_EQ(deterministic_random.discrete_log_index(deterministic), 1U);
  }

  const std::array shifted_log_weights{-10'000.0, -10'001.0, negative_infinity};
  const std::array ordinary_weights{1.0, std::exp(-1.0), 0.0};
  qmc::Random shifted_random(772);
  qmc::Random ordinary_random(772);
  for (int draw = 0; draw < 200; ++draw) {
    EXPECT_EQ(shifted_random.discrete_log_index(shifted_log_weights),
              ordinary_random.discrete_index(ordinary_weights));
  }
}

} // namespace
