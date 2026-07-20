#include "adaptive_discrete_support.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <stdexcept>

namespace qmc::detail {
namespace {

AdaptiveDiscreteSupport make_support(const std::size_t initial, const std::size_t maximum,
                                     const double tolerance = 1e-3) {
  return AdaptiveDiscreteSupport(initial, maximum, tolerance, "support limit reached",
                                 "support overflowed", "support mass is invalid");
}

TEST(AdaptiveDiscreteSupportTest, AccumulatesIncludedMassAndAppliesRelativeTailTolerance) {
  AdaptiveDiscreteSupport support = make_support(4, 20);
  support.add_weight(2.0, 2.0);
  support.add_log_weight(std::log(1.0));

  EXPECT_NEAR(std::exp(support.log_included_mass()), 5.0, 1e-14);
  EXPECT_FALSE(support.tail_is_controlled(std::nullopt));
  EXPECT_FALSE(support.tail_is_controlled(std::log(0.006)));
  EXPECT_TRUE(support.tail_is_controlled(std::log(0.004)));
  EXPECT_TRUE(support.tail_is_controlled(-std::numeric_limits<double>::infinity()));
}

TEST(AdaptiveDiscreteSupportTest, GrowsDeterministicallyAndEnforcesTheHardLimit) {
  AdaptiveDiscreteSupport support = make_support(4, 20);
  support.grow(SupportGrowth{.minimum = 8, .factor = 1.5});
  EXPECT_EQ(support.support(), 12U);
  support.grow(SupportGrowth{.minimum = 8, .factor = 1.5});
  EXPECT_EQ(support.support(), 20U);
  EXPECT_THROW(support.grow(SupportGrowth{.minimum = 8, .factor = 1.5}), std::runtime_error);
  EXPECT_EQ(support.support(), 20U);
}

TEST(AdaptiveDiscreteSupportTest, RejectsInvalidMassAndTailChecksWithoutMass) {
  AdaptiveDiscreteSupport empty = make_support(4, 20);
  EXPECT_THROW(static_cast<void>(empty.tail_is_controlled(0.0)), std::runtime_error);
  EXPECT_THROW(empty.add_weight(-1.0), std::runtime_error);
  EXPECT_THROW(empty.add_log_weight(std::numeric_limits<double>::infinity()), std::runtime_error);

  EXPECT_THROW(static_cast<void>(make_support(21, 20)), std::runtime_error);
  EXPECT_THROW(static_cast<void>(make_support(4, 20, 1.0)), std::invalid_argument);
  EXPECT_THROW(empty.grow(SupportGrowth{.minimum = 0, .factor = 1.5}), std::logic_error);
}

} // namespace
} // namespace qmc::detail
