#include "qmc/configuration.hpp"

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

bool samples_are_equal(const qmc::IdealBosonConfiguration &left,
                       const qmc::IdealBosonConfiguration &right) {
  if (left.permutation != right.permutation || left.worldlines.values != right.worldlines.values ||
      left.worldlines_covering.values != right.worldlines_covering.values ||
      left.cycles.size() != right.cycles.size() || left.log_ZN != right.log_ZN) {
    return false;
  }
  for (std::size_t index = 0; index < left.cycles.size(); ++index) {
    const auto &left_cycle = left.cycles[index];
    const auto &right_cycle = right.cycles[index];
    if (left_cycle.labels != right_cycle.labels ||
        left_cycle.base_point != right_cycle.base_point ||
        left_cycle.winding != right_cycle.winding ||
        left_cycle.covering_path != right_cycle.covering_path ||
        left_cycle.torus_path != right_cycle.torus_path) {
      return false;
    }
  }
  return true;
}

TEST(DenseWorldlinesTest, StoresFlatRowMajorValuesWithBoundsChecking) {
  qmc::DenseWorldlines lines(2, 3, 2);
  lines.at(1, 2, 1) = 17;
  EXPECT_EQ(lines.at(1, 2, 1), 17);
  EXPECT_EQ(lines.values.size(), 12U);
  EXPECT_THROW(static_cast<void>(lines.at(2, 0, 0)), std::out_of_range);
  lines.values.pop_back();
  EXPECT_THROW(lines.validate_shape(), std::logic_error);
}

TEST(IdealConfigurationTest, AssemblesConsistentWorldLines) {
  const qmc::Model model{
      .particle_count = 7,
      .beta = 1.2,
      .linear_size = 6,
      .dimension = 2,
      .hopping = 0.9,
  };
  qmc::Random random(2026);
  const auto configuration = qmc::sample_ideal_boson_configuration(model, 7, random);
  EXPECT_NO_THROW(configuration.validate());
  EXPECT_EQ(configuration.worldlines.particles, 7U);
  EXPECT_EQ(configuration.worldlines.time_points, 8U);
  EXPECT_EQ(configuration.worldlines.dimension, 2U);

  std::vector<qmc::ParticleId> labels;
  for (const auto &cycle : configuration.cycles) {
    labels.insert(labels.end(), cycle.labels.begin(), cycle.labels.end());
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      EXPECT_EQ(cycle.covering_path.back()[axis] - cycle.covering_path.front()[axis],
                model.linear_size * cycle.winding[axis]);
      EXPECT_EQ(cycle.torus_path.front()[axis], cycle.torus_path.back()[axis]);
    }
  }
  std::ranges::sort(labels);
  for (std::size_t index = 0; index < labels.size(); ++index) {
    EXPECT_EQ(labels[index], index);
  }
}

TEST(IdealConfigurationTest, SupportsEmptySystem) {
  const qmc::Model model{
      .particle_count = 0,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 2,
      .hopping = 1.0,
  };
  qmc::Random random(3);
  const auto configuration = qmc::sample_ideal_boson_configuration(model, 5, random);
  EXPECT_TRUE(configuration.cycles.empty());
  EXPECT_TRUE(configuration.permutation.empty());
  EXPECT_TRUE(configuration.worldlines.values.empty());
  EXPECT_DOUBLE_EQ(configuration.log_ZN, 0.0);
  EXPECT_NO_THROW(configuration.validate());
}

TEST(IdealConfigurationTest, ZeroHoppingProducesConstantCoveringPaths) {
  const qmc::Model model{
      .particle_count = 8,
      .beta = 2.0,
      .linear_size = 5,
      .dimension = 3,
      .hopping = 0.0,
  };
  qmc::Random random(91);
  const auto configuration = qmc::sample_ideal_boson_configuration(model, 9, random);
  for (const auto &cycle : configuration.cycles) {
    EXPECT_TRUE(
        std::ranges::all_of(cycle.winding, [](const qmc::Coord value) { return value == 0; }));
    EXPECT_TRUE(std::ranges::all_of(cycle.covering_path, [&cycle](const qmc::Site &site) {
      return site == cycle.covering_path.front();
    }));
  }
}

TEST(IdealConfigurationTest, ValidationDetectsPermutationCorruption) {
  const qmc::Model model{
      .particle_count = 4,
      .beta = 1.0,
      .linear_size = 6,
      .dimension = 1,
      .hopping = 1.0,
  };
  qmc::Random random(7);
  auto configuration = qmc::sample_ideal_boson_configuration(model, 4, random);
  configuration.permutation[0] = configuration.permutation[1];
  EXPECT_THROW(configuration.validate(), std::logic_error);
}

TEST(IdealConfigurationTest, ReusableEnsembleMatchesOneOffWrapperForTheSameSeed) {
  const qmc::Model model{
      .particle_count = 6,
      .beta = 0.9,
      .linear_size = 8,
      .dimension = 2,
      .hopping = 1.1,
  };
  const qmc::CanonicalEnsemble ensemble(model);
  qmc::Random first_random(8128);
  qmc::Random second_random(8128);
  const auto first = qmc::sample_ideal_boson_configuration(model, 11, first_random);
  const auto second = qmc::sample_ideal_boson_configuration(ensemble, 11, second_random);

  EXPECT_TRUE(samples_are_equal(first, second));
}

TEST(IdealConfigurationTest, RepeatedSamplingHandlesNegativeUnitWinding) {
  const qmc::Model model{
      .particle_count = 4,
      .beta = 0.8,
      .linear_size = 5,
      .dimension = 2,
      .hopping = 0.9,
  };
  qmc::Random random(2026);
  for (std::size_t sample = 0; sample < 20; ++sample) {
    EXPECT_NO_THROW(static_cast<void>(qmc::sample_ideal_boson_configuration(model, 8, random)));
  }
}

} // namespace
