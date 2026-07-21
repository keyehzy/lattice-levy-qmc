#include "qmc/model.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

static_assert(!std::is_default_constructible_v<qmc::Model>);
static_assert(!std::is_aggregate_v<qmc::Model>);
static_assert(noexcept(std::declval<const qmc::Model &>().volume()));

TEST(ModelTest, ValidatesAndComputesVolume) {
  const qmc::Model model(qmc::ModelParameters{
      .particle_count = 7,
      .beta = 1.2,
      .linear_size = 6,
      .dimension = 3,
      .hopping = 0.9,
  });
  EXPECT_EQ(model.particle_count(), 7U);
  EXPECT_DOUBLE_EQ(model.beta(), 1.2);
  EXPECT_EQ(model.linear_size(), 6);
  EXPECT_EQ(model.dimension(), 3U);
  EXPECT_DOUBLE_EQ(model.hopping(), 0.9);
  EXPECT_EQ(model.volume(), 216U);
}

TEST(ModelTest, RejectsInvalidParameters) {
  EXPECT_THROW(static_cast<void>(qmc::Model(qmc::ModelParameters{.beta = -1.0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::Model(qmc::ModelParameters{.linear_size = 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::Model(qmc::ModelParameters{.dimension = 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::Model(
                   qmc::ModelParameters{.hopping = std::numeric_limits<double>::infinity()})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(qmc::Model(qmc::ModelParameters{
                   .particle_count =
                       static_cast<std::size_t>(std::numeric_limits<qmc::ParticleId>::max()) + 1})),
               std::invalid_argument);
}

TEST(ModelTest, RejectsVolumeOverflowAtConstruction) {
  EXPECT_THROW(static_cast<void>(qmc::Model(qmc::ModelParameters{
                   .linear_size = std::numeric_limits<qmc::Coord>::max(),
                   .dimension = 3,
               })),
               std::overflow_error);
}

TEST(NumericalOptionsTest, RejectsInvalidLimits) {
  qmc::NumericalOptions options;
  EXPECT_NO_THROW(options.validate());
  options.tail_tolerance = 1.0;
  EXPECT_THROW(options.validate(), std::invalid_argument);
  options.tail_tolerance = 1e-14;
  options.max_bessel_terms = 0;
  EXPECT_THROW(options.validate(), std::invalid_argument);
  options.max_bessel_terms = 1;
  options.max_winding = 0;
  EXPECT_THROW(options.validate(), std::invalid_argument);
}

} // namespace
