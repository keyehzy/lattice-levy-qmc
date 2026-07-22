#include "qmc/matsubara_modes.hpp"
#include "qmc/observables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace qmc {
namespace {

static_assert(!std::default_initializable<MatsubaraModeSet>);
static_assert(!std::default_initializable<MatsubaraModeField<double>>);
static_assert(std::is_same_v<decltype(std::declval<const MatsubaraModeSet &>().layout()),
                             const TorusLayout &>);
static_assert(std::is_same_v<decltype(std::declval<const MatsubaraModeField<double> &>().values()),
                             std::span<const double>>);

MatsubaraModeSet make_modes(const double beta, const TorusLayout &layout,
                            std::vector<std::vector<std::size_t>> momenta,
                            std::vector<std::int64_t> frequencies) {
  return MatsubaraModeSet(beta, layout,
                          MatsubaraModeRequest{.momentum_indices = std::move(momenta),
                                               .frequency_indices = std::move(frequencies)});
}

TEST(MatsubaraModeSetTest, PreservesSignedRequestOrderAndDerivedGeometry) {
  const MatsubaraModeSet modes = make_modes(2.0, TorusLayout(4, 2), {{3, 0}, {0, 2}}, {2, -1, 0});

  EXPECT_DOUBLE_EQ(modes.beta(), 2.0);
  EXPECT_EQ(modes.layout(), TorusLayout(4, 2));
  EXPECT_EQ(modes.momentum_count(), 2U);
  EXPECT_EQ(modes.frequency_count(), 3U);
  EXPECT_EQ(modes.mode_count(), 6U);
  EXPECT_TRUE(std::ranges::equal(modes.momentum_indices(0), std::array<std::size_t, 2>{3, 0}));
  EXPECT_TRUE(std::ranges::equal(modes.momentum_indices(1), std::array<std::size_t, 2>{0, 2}));
  EXPECT_NEAR(modes.wavevector_component(0, 0), 1.5 * std::numbers::pi, 1e-15);
  EXPECT_DOUBLE_EQ(modes.wavevector_component(0, 1), 0.0);
  EXPECT_NEAR(modes.wavevector_component(1, 1), std::numbers::pi, 1e-15);
  EXPECT_EQ(modes.frequency_index(0), 2);
  EXPECT_EQ(modes.frequency_index(1), -1);
  EXPECT_EQ(modes.frequency_index(2), 0);
  EXPECT_NEAR(modes.frequency(0), 2.0 * std::numbers::pi, 1e-15);
  EXPECT_NEAR(modes.frequency(1), -std::numbers::pi, 1e-15);
  EXPECT_DOUBLE_EQ(modes.frequency(2), 0.0);

  EXPECT_THROW(static_cast<void>(modes.momentum_indices(2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(modes.wavevector_component(0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(modes.frequency_index(3)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(modes.frequency(3)), std::out_of_range);
}

TEST(MatsubaraModeSetTest, RejectsInvalidBetaAndEmptyRequests) {
  const TorusLayout layout(3, 1);
  EXPECT_THROW(static_cast<void>(make_modes(0.0, layout, {{0}}, {0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_modes(-1.0, layout, {{0}}, {0})), std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(make_modes(std::numeric_limits<double>::infinity(), layout, {{0}}, {0})),
      std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(make_modes(std::numeric_limits<double>::quiet_NaN(), layout, {{0}}, {0})),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_modes(1.0, layout, {}, {0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_modes(1.0, layout, {{0}}, {})), std::invalid_argument);
}

TEST(MatsubaraModeSetTest, RejectsMalformedOrDuplicateModeIdentities) {
  const TorusLayout layout(3, 2);
  EXPECT_THROW(static_cast<void>(make_modes(1.0, layout, {{0}}, {0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_modes(1.0, layout, {{0, 3}}, {0})), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_modes(1.0, layout, {{1, 2}, {1, 2}}, {0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(make_modes(1.0, layout, {{1, 2}}, {-4, -4})),
               std::invalid_argument);
}

TEST(MatsubaraModeSetTest, SeparatesRepresentabilityFromContinuousPlanLimit) {
  const TorusLayout layout(2, 1);
  const MatsubaraModeSet wide = make_modes(
      1.0, layout, {{0}},
      {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()});
  EXPECT_TRUE(std::isfinite(wide.frequency(0)));
  EXPECT_TRUE(std::isfinite(wide.frequency(1)));
  EXPECT_LT(wide.frequency(0), 0.0);
  EXPECT_GT(wide.frequency(1), 0.0);

  EXPECT_THROW(
      static_cast<void>(make_modes(std::numeric_limits<double>::denorm_min(), layout, {{0}}, {1})),
      std::overflow_error);
}

TEST(MatsubaraModeFieldTest, OwnsFrequencyMajorValuesAndChecksEveryAxis) {
  const MatsubaraModeSet modes = make_modes(1.5, TorusLayout(3, 1), {{2}, {0}}, {-1, 3});
  const MatsubaraModeField<int> field(modes, {10, 11, 20, 21});

  EXPECT_EQ(field.modes(), modes);
  EXPECT_TRUE(std::ranges::equal(field.values(), std::array{10, 11, 20, 21}));
  EXPECT_EQ(field.at(0, 0), 10);
  EXPECT_EQ(field.at(0, 1), 11);
  EXPECT_EQ(field.at(1, 0), 20);
  EXPECT_EQ(field.at(1, 1), 21);
  EXPECT_THROW(static_cast<void>(field.at(2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(field.at(0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(MatsubaraModeField<int>(modes, {1, 2, 3})), std::invalid_argument);
}

TEST(MatsubaraDensityCorrelationsTest, RequiresExactRetainedGridModeProvenance) {
  const RetainedGrid grid(2.0, TorusLayout(2, 1), 2);
  const MatsubaraModeSet modes = make_modes(2.0, grid.layout(), {{0}, {1}}, {0, 1});
  const MatsubaraDensityCorrelations valid(
      grid, MatsubaraModeField<std::complex<double>>(
                modes, {{1.0, 0.0}, {2.0, 1.0}, {3.0, -1.0}, {4.0, 0.0}}));

  EXPECT_EQ(valid.grid(), grid);
  EXPECT_EQ(valid.modes(), modes);
  EXPECT_EQ(valid.values().size(), 4U);
  EXPECT_EQ(valid.at(1, 0), std::complex<double>(3.0, -1.0));
  EXPECT_THROW(static_cast<void>(valid.at(2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(valid.at(0, 2)), std::out_of_range);

  const auto field = [](MatsubaraModeSet field_modes) {
    const std::size_t value_count = field_modes.mode_count();
    return MatsubaraModeField<std::complex<double>>(std::move(field_modes),
                                                    std::vector<std::complex<double>>(value_count));
  };
  EXPECT_THROW(static_cast<void>(MatsubaraDensityCorrelations(
                   grid, field(make_modes(1.0, grid.layout(), {{0}, {1}}, {0, 1})))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(MatsubaraDensityCorrelations(
                   grid, field(make_modes(2.0, TorusLayout(4, 1), {{0}, {1}, {2}, {3}}, {0, 1})))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(MatsubaraDensityCorrelations(
                   grid, field(make_modes(2.0, grid.layout(), {{0}, {1}}, {1, 0})))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(MatsubaraDensityCorrelations(
                   grid, field(make_modes(2.0, grid.layout(), {{1}, {0}}, {0, 1})))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(MatsubaraDensityCorrelations(
                   grid, field(make_modes(2.0, grid.layout(), {{0}}, {0, 1})))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(MatsubaraDensityCorrelations(
                   grid, field(make_modes(2.0, grid.layout(), {{0}, {1}}, {0})))),
               std::invalid_argument);

  const RetainedGrid line_grid(2.0, TorusLayout(4, 1), 1);
  EXPECT_THROW(static_cast<void>(MatsubaraDensityCorrelations(
                   line_grid, field(make_modes(2.0, TorusLayout(2, 2),
                                               {{0, 0}, {1, 0}, {0, 1}, {1, 1}}, {0})))),
               std::invalid_argument);

  if constexpr (std::numeric_limits<std::size_t>::max() >
                static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    const auto unrepresentable_count =
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) + 2U;
    const RetainedGrid unrepresentable_grid(2.0, TorusLayout(1, 1), unrepresentable_count);
    EXPECT_THROW(static_cast<void>(MatsubaraDensityCorrelations(
                     unrepresentable_grid, field(make_modes(2.0, TorusLayout(1, 1), {{0}}, {0})))),
                 std::overflow_error);
  }
}

TEST(MatsubaraDensityCorrelationsTest, TransformPublishesFullFlatModeOrdering) {
  const RetainedGrid grid(2.0, TorusLayout(2, 2), 2);
  const ImaginaryTimeDensityCorrelations correlations(grid,
                                                      {0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0});
  const MatsubaraDensityCorrelations transformed = retained_grid_matsubara_transform(correlations);

  EXPECT_EQ(transformed.grid(), grid);
  EXPECT_EQ(transformed.modes().momentum_count(), grid.layout().volume());
  EXPECT_EQ(transformed.modes().frequency_count(), grid.time_points());
  EXPECT_EQ(transformed.values().size(), 8U);
  for (std::size_t momentum = 0; momentum < grid.layout().volume(); ++momentum) {
    EXPECT_TRUE(std::ranges::equal(transformed.modes().momentum_indices(momentum),
                                   grid.layout().decode(SiteId(momentum))));
  }
  for (std::size_t frequency = 0; frequency < grid.time_points(); ++frequency) {
    EXPECT_EQ(transformed.modes().frequency_index(frequency), static_cast<std::int64_t>(frequency));
  }
  EXPECT_NEAR(transformed.modes().frequency(1), std::numbers::pi, 1e-14);
  EXPECT_EQ(transformed.at(0, 0), transformed.values()[0]);
  EXPECT_EQ(transformed.at(1, 3), transformed.values()[7]);
  const std::array expected_real{127.0, -43.0, -77.0, 25.0, -113.0, 37.0, 67.0, -23.0};
  for (std::size_t mode = 0; mode < expected_real.size(); ++mode) {
    EXPECT_NEAR(transformed.values()[mode].real(), expected_real[mode], 2e-14);
    EXPECT_NEAR(transformed.values()[mode].imag(), 0.0, 2e-14);
  }
}

} // namespace
} // namespace qmc
