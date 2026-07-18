#include "qmc/model.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {

TEST(ModelTest, ValidatesAndComputesVolume) {
  const qmc::Model model{
      .particle_count = 7,
      .beta = 1.2,
      .linear_size = 6,
      .dimension = 3,
      .hopping = 0.9,
  };
  EXPECT_NO_THROW(model.validate());
  EXPECT_EQ(model.volume(), 216U);
}

TEST(ModelTest, RejectsInvalidParameters) {
  qmc::Model model;
  model.beta = -1.0;
  EXPECT_THROW(model.validate(), std::invalid_argument);
  model.beta = 0.0;
  model.linear_size = 0;
  EXPECT_THROW(model.validate(), std::invalid_argument);
  model.linear_size = 1;
  model.dimension = 0;
  EXPECT_THROW(model.validate(), std::invalid_argument);
  model.dimension = 1;
  model.hopping = std::numeric_limits<double>::infinity();
  EXPECT_THROW(model.validate(), std::invalid_argument);
}

TEST(ModelTest, DetectsVolumeOverflow) {
  const qmc::Model model{
      .particle_count = 0,
      .beta = 0.0,
      .linear_size = std::numeric_limits<qmc::Coord>::max(),
      .dimension = 3,
      .hopping = 0.0,
  };
  EXPECT_THROW(static_cast<void>(model.volume()), std::overflow_error);
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
