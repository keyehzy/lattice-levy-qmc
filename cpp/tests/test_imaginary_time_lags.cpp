#include "qmc/imaginary_time_lags.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace qmc {
namespace {

static_assert(!std::default_initializable<ImaginaryTimeLagSet>);
static_assert(std::is_same_v<decltype(std::declval<const ImaginaryTimeLagSet &>().layout()),
                             const TorusLayout &>);

ImaginaryTimeLagSet make_lags(const double beta, const TorusLayout &layout,
                              std::vector<std::vector<std::size_t>> momenta,
                              std::vector<double> lags) {
  return ImaginaryTimeLagSet(
      beta, layout,
      ImaginaryTimeLagRequest{.momentum_indices = std::move(momenta), .lags = std::move(lags)});
}

TEST(ImaginaryTimeLagSetTest, PreservesRequestOrderAndDerivedGeometry) {
  const ImaginaryTimeLagSet lags =
      make_lags(2.0, TorusLayout(4, 2), {{3, 0}, {0, 2}}, {1.5, 0.0, 0.25});

  EXPECT_DOUBLE_EQ(lags.beta(), 2.0);
  EXPECT_EQ(lags.layout(), TorusLayout(4, 2));
  EXPECT_EQ(lags.momentum_count(), 2U);
  EXPECT_EQ(lags.lag_count(), 3U);
  EXPECT_EQ(lags.value_count(), 6U);
  EXPECT_TRUE(std::ranges::equal(lags.momentum_indices(0), std::array<std::size_t, 2>{3, 0}));
  EXPECT_TRUE(std::ranges::equal(lags.momentum_indices(1), std::array<std::size_t, 2>{0, 2}));
  EXPECT_NEAR(lags.wavevector_component(0, 0), 1.5 * std::numbers::pi, 1e-15);
  EXPECT_DOUBLE_EQ(lags.wavevector_component(0, 1), 0.0);
  EXPECT_NEAR(lags.wavevector_component(1, 1), std::numbers::pi, 1e-15);
  EXPECT_DOUBLE_EQ(lags.lag(0), 1.5);
  EXPECT_DOUBLE_EQ(lags.lag(1), 0.0);
  EXPECT_DOUBLE_EQ(lags.lag(2), 0.25);

  EXPECT_THROW(static_cast<void>(lags.momentum_indices(2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(lags.wavevector_component(0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(lags.lag(3)), std::out_of_range);
}

TEST(ImaginaryTimeLagSetTest, RejectsInvalidBetaAndEmptyRequests) {
  const TorusLayout layout(3, 1);
  EXPECT_THROW(static_cast<void>(make_lags(0.0, layout, {{0}}, {0.0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_lags(-1.0, layout, {{0}}, {0.0})), std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(make_lags(std::numeric_limits<double>::infinity(), layout, {{0}}, {0.0})),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(make_lags(std::numeric_limits<double>::quiet_NaN(), layout, {{0}}, {0.0})),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_lags(1.0, layout, {}, {0.0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_lags(1.0, layout, {{0}}, {})), std::invalid_argument);
}

TEST(ImaginaryTimeLagSetTest, RejectsMalformedOrDuplicateMomentaAndLags) {
  const TorusLayout layout(3, 2);
  EXPECT_THROW(static_cast<void>(make_lags(1.0, layout, {{0}}, {0.0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_lags(1.0, layout, {{0, 3}}, {0.0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_lags(1.0, layout, {{1, 2}, {1, 2}}, {0.0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_lags(1.0, layout, {{1, 2}}, {0.25, 0.25})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_lags(1.0, layout, {{1, 2}}, {0.0, -0.0})),
               std::invalid_argument);
}

TEST(ImaginaryTimeLagSetTest, RequiresFiniteCanonicalLags) {
  const TorusLayout layout(2, 1);
  EXPECT_THROW(static_cast<void>(make_lags(1.0, layout, {{0}}, {-0.1})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_lags(1.0, layout, {{0}}, {1.0})), std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(make_lags(1.0, layout, {{0}}, {std::numeric_limits<double>::infinity()})),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(make_lags(1.0, layout, {{0}}, {std::numeric_limits<double>::quiet_NaN()})),
      std::invalid_argument);

  const ImaginaryTimeLagSet negative_zero = make_lags(1.0, layout, {{0}}, {-0.0});
  EXPECT_DOUBLE_EQ(negative_zero.lag(0), 0.0);
  EXPECT_FALSE(std::signbit(negative_zero.lag(0)));
}

} // namespace
} // namespace qmc
