#include "qmc/free_numerics.hpp"
#include "torus_bridge_distribution.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

TEST(FreeNumericsTest, UsesEuclideanModulo) {
  EXPECT_EQ(qmc::torus_mod(-13, 6), 5);
  EXPECT_EQ(qmc::torus_mod(-6, 6), 0);
  EXPECT_EQ(qmc::torus_mod(8, 6), 2);
  EXPECT_THROW(static_cast<void>(qmc::torus_mod(0, 0)), std::invalid_argument);
}

TEST(FreeNumericsTest, LogSumExpHandlesLargeAndInfiniteInputs) {
  const std::array values{1000.0, 999.0, -std::numeric_limits<double>::infinity()};
  EXPECT_NEAR(qmc::log_sum_exp(values), 1000.0 + std::log1p(std::exp(-1.0)), 2e-13);

  const std::array negative_infinity{-std::numeric_limits<double>::infinity()};
  EXPECT_EQ(qmc::log_sum_exp(negative_infinity), -std::numeric_limits<double>::infinity());
  EXPECT_THROW(static_cast<void>(qmc::log_sum_exp(std::span<const double>{})),
               std::invalid_argument);
}

TEST(FreeNumericsTest, ScaledBesselSatisfiesConvolution) {
  constexpr qmc::Coord a = 0;
  constexpr qmc::Coord b = 3;
  constexpr double left_argument = 1.3;
  constexpr double right_argument = 2.1;
  double convolution = 0.0;
  for (qmc::Coord midpoint = -80; midpoint <= 80; ++midpoint) {
    const auto left_order = static_cast<std::uint64_t>(std::abs(midpoint - a));
    const auto right_order = static_cast<std::uint64_t>(std::abs(b - midpoint));
    convolution += qmc::scaled_modified_bessel_i(left_order, left_argument) *
                   qmc::scaled_modified_bessel_i(right_order, right_argument);
  }
  const double direct = qmc::scaled_modified_bessel_i(static_cast<std::uint64_t>(std::abs(b - a)),
                                                      left_argument + right_argument);
  EXPECT_NEAR(convolution, direct, 5e-14);
}

TEST(FreeNumericsTest, ScaledBesselReturnsUnderflowInsteadOfAbortingProcess) {
  EXPECT_DOUBLE_EQ(qmc::scaled_modified_bessel_i(10'000, 4.0), 0.0);
}

TEST(FreePathKernelsTest, OwnsValidatedNumericsAndOptionalTorusProvenance) {
  const qmc::NumericalOptions numerical{
      .tail_tolerance = 1e-12,
      .max_bessel_terms = 20'000,
      .max_winding = 10'000,
  };
  const qmc::Model model(qmc::ModelParameters{
      .particle_count = 3,
      .beta = 0.8,
      .linear_size = 7,
      .dimension = 2,
      .hopping = 0.9,
  });
  const qmc::FreePathKernels covering_only(model.hopping(), numerical);
  const qmc::FreePathKernels torus(model, numerical);

  EXPECT_DOUBLE_EQ(torus.hopping(), model.hopping());
  EXPECT_EQ(torus.numerical(), numerical);
  ASSERT_NE(torus.torus_layout(), nullptr);
  EXPECT_EQ(*torus.torus_layout(), qmc::TorusLayout(model.linear_size(), model.dimension()));
  EXPECT_EQ(covering_only.torus_layout(), nullptr);
  EXPECT_THROW(static_cast<void>(covering_only.periodic_kernel_scaled_1d(0, 1.0)),
               std::invalid_argument);

  EXPECT_THROW(static_cast<void>(qmc::FreePathKernels(-0.1)), std::invalid_argument);
  qmc::NumericalOptions invalid = numerical;
  invalid.max_bessel_terms = 0;
  EXPECT_THROW(static_cast<void>(qmc::FreePathKernels(model, invalid)), std::invalid_argument);
}

TEST(FreePathKernelsTest, CoveringMethodsMatchOneOffWrappersAndRandomStream) {
  const qmc::NumericalOptions numerical{
      .tail_tolerance = 1e-12,
      .max_bessel_terms = 20'000,
      .max_winding = 10'000,
  };
  const qmc::FreePathKernels kernels(0.8, numerical);
  qmc::Random context_random(9173);
  qmc::Random wrapper_random(9173);

  EXPECT_EQ(kernels.sample_bessel_pair_count(3, 1.7, context_random),
            qmc::sample_bessel_pair_count(3, 1.7, wrapper_random, numerical));
  EXPECT_EQ(
      kernels.sample_midpoint_covering({0, -2}, {4, 3}, 0.4, 0.9, context_random),
      qmc::sample_midpoint_covering({0, -2}, {4, 3}, 0.4, 0.9, 0.8, wrapper_random, numerical));
  EXPECT_EQ(kernels.sample_bridge_covering({1, -1}, {5, 2}, 1.3, 13, context_random),
            qmc::sample_bridge_covering({1, -1}, {5, 2}, 1.3, 13, 0.8, wrapper_random, numerical));
  EXPECT_DOUBLE_EQ(context_random.uniform_open(), wrapper_random.uniform_open());
}

TEST(FreePathKernelsTest, PreparedTorusBridgeReusesNormalizationAndSeededWindingLaw) {
  const qmc::Model model(qmc::ModelParameters{
      .particle_count = 3,
      .beta = 1.2,
      .linear_size = 6,
      .dimension = 2,
      .hopping = 1.1,
  });
  const qmc::FreePathKernels kernels(model);
  const qmc::TorusLayout &layout = *kernels.torus_layout();
  const qmc::Site start{7, -3};
  const qmc::Site physical_end{1, 4};
  constexpr double duration = 0.7;
  const qmc::SiteId displacement =
      layout.flat_displacement(layout.encode_covering(start), layout.encode_covering(physical_end));
  const qmc::detail::TorusBridgeDistribution prepared(displacement, duration, kernels);

  const double expected_log_normalization =
      std::log(kernels.periodic_kernel_scaled_1d(0, duration)) +
      std::log(kernels.periodic_kernel_scaled_1d(1, duration));
  EXPECT_NEAR(prepared.log_normalization(), expected_log_normalization, 2e-14);
  EXPECT_DOUBLE_EQ(prepared.log_normalization(),
                   kernels.log_torus_kernel_scaled(start, physical_end, duration));

  qmc::Random prepared_random(27'104);
  qmc::Random one_off_random(27'104);
  for (std::size_t sample = 0; sample < 100; ++sample) {
    EXPECT_EQ(
        prepared.sample_covering_endpoint(start, prepared_random),
        kernels.sample_torus_covering_endpoint(start, physical_end, duration, one_off_random));
  }
  EXPECT_DOUBLE_EQ(prepared_random.uniform_open(), one_off_random.uniform_open());
}

TEST(FreePathKernelsTest, PreparedTorusBridgeRejectsZeroWeightWithoutDrawing) {
  const qmc::FreePathKernels kernels(qmc::TorusLayout(5, 1), 0.0);
  const qmc::detail::TorusBridgeDistribution prepared(qmc::SiteId(1), 0.8, kernels);
  EXPECT_EQ(prepared.log_normalization(), -std::numeric_limits<double>::infinity());

  qmc::Random random(92);
  qmc::Random control(92);
  EXPECT_THROW(static_cast<void>(prepared.sample_covering_endpoint({0}, random)),
               std::invalid_argument);
  EXPECT_DOUBLE_EQ(random.uniform_unit(), control.uniform_unit());
  EXPECT_THROW(static_cast<void>(prepared.sample_covering_endpoint({0, 0}, random)),
               std::invalid_argument);
}

TEST(FreeNumericsTest, ExactMidpointPmfMatchesPythonReference) {
  std::vector<qmc::Coord> coordinates;
  for (qmc::Coord coordinate = -8; coordinate <= 10; ++coordinate) {
    coordinates.push_back(coordinate);
  }
  const auto probabilities = qmc::exact_midpoint_pmf_window(0, 2, 0.5, 0.5, 1.0, coordinates);
  const std::array reference{
      3.9801353061749075e-17, 1.2809526326684644e-14, 3.2513478001972921e-12,
      6.3013252959034777e-10, 8.9345103075541945e-08, 8.7349214772972949e-06,
      5.3931131294431326e-04, 1.8185231723809047e-02, 2.4946059933870135e-01,
      4.6361206544913652e-01, 2.4946059933870135e-01, 1.8185231723809047e-02,
      5.3931131294431326e-04, 8.7349214772972949e-06, 8.9345103075541945e-08,
      6.3013252959034777e-10, 3.2513478001972921e-12, 1.2809526326684644e-14,
      3.9801353061749075e-17,
  };
  ASSERT_EQ(probabilities.size(), reference.size());
  for (std::size_t index = 0; index < reference.size(); ++index) {
    EXPECT_NEAR(probabilities[index], reference[index], 3e-15);
  }
  EXPECT_NEAR(std::accumulate(probabilities.begin(), probabilities.end(), 0.0), 1.0, 2e-14);
}

TEST(FreeNumericsTest, MidpointHandlesDegenerateDurations) {
  qmc::Random random(42);
  EXPECT_EQ(qmc::sample_midpoint_covering_1d(3, 3, 0.0, 0.0, 1.0, random), 3);
  EXPECT_EQ(qmc::sample_midpoint_covering_1d(3, 9, 0.0, 1.0, 1.0, random), 3);
  EXPECT_EQ(qmc::sample_midpoint_covering_1d(3, 9, 1.0, 0.0, 1.0, random), 9);
  EXPECT_THROW(static_cast<void>(qmc::sample_midpoint_covering_1d(3, 9, 0.0, 0.0, 1.0, random)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::sample_midpoint_covering_1d(3, 9, 0.5, 0.5, 0.0, random)),
               std::invalid_argument);
}

TEST(FreeNumericsTest, BridgeSupportsNonPowerOfTwoGrid) {
  qmc::Random random(123);
  const qmc::Site start{0, 0};
  const qmc::Site end{3, -2};
  const auto path = qmc::sample_bridge_covering(start, end, 1.7, 15, 0.8, random);
  ASSERT_EQ(path.size(), 16U);
  EXPECT_EQ(path.front(), start);
  EXPECT_EQ(path.back(), end);
  for (const auto &site : path) {
    EXPECT_EQ(site.size(), 2U);
  }
}

TEST(FreeNumericsTest, BridgeRejectsImpossibleOrMalformedInputs) {
  qmc::Random random(1);
  EXPECT_THROW(static_cast<void>(qmc::sample_bridge_covering({0}, {0}, 1.0, 0, 1.0, random)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::sample_bridge_covering({0}, {0, 1}, 1.0, 1, 1.0, random)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::sample_bridge_covering({0}, {1}, 0.0, 1, 1.0, random)),
               std::invalid_argument);
}

TEST(FreeNumericsTest, PeriodicKernelMatchesPythonReference) {
  const std::array reference{0.3286011450576396, 0.22311963012458955, 0.11257979734659065,
                             0.11257979734659067, 0.2231196301245896};
  for (std::size_t delta = 0; delta < reference.size(); ++delta) {
    EXPECT_NEAR(qmc::periodic_kernel_scaled_1d(static_cast<qmc::Coord>(delta), 1.3, 5, 0.7),
                reference[delta], 5e-16);
  }
}

TEST(FreeNumericsTest, BesselCountHonorsWorkLimit) {
  qmc::Random random(2);
  const qmc::NumericalOptions options{
      .tail_tolerance = 1e-14,
      .max_bessel_terms = 2,
      .max_winding = 10,
  };
  EXPECT_THROW(static_cast<void>(qmc::sample_bessel_pair_count(0, 10.0, random, options)),
               std::runtime_error);
}

} // namespace
