#ifndef QMC_CONFIGURATION_HPP
#define QMC_CONFIGURATION_HPP

#include "qmc/free_boson.hpp"
#include "qmc/free_numerics.hpp"
#include "qmc/model.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <vector>

namespace qmc {

struct DenseWorldlines {
  std::size_t particles = 0;
  std::size_t time_points = 0;
  std::size_t dimension = 0;
  std::vector<Coord> values;

  DenseWorldlines() = default;
  DenseWorldlines(std::size_t particle_count, std::size_t point_count, std::size_t dimensions);

  // Bounds-checked row-major access in particle, time, axis order.
  [[nodiscard]] Coord &at(ParticleId particle, std::size_t time_index, std::size_t axis);
  [[nodiscard]] const Coord &at(ParticleId particle, std::size_t time_index,
                                std::size_t axis) const;
  void validate_shape() const;
};

struct IdealCyclePath {
  Cycle labels;
  Site base_point;
  Site winding;
  CoveringPath covering_path;
  CoveringPath torus_path;
};

struct IdealBosonConfiguration {
  Model model;
  std::size_t time_links_per_beta = 1;
  std::vector<IdealCyclePath> cycles;
  std::vector<ParticleId> permutation;
  DenseWorldlines worldlines;
  DenseWorldlines worldlines_covering;
  double log_ZN = 0.0;

  // Checks shapes, cycle partitioning, permutation endpoints, modulo reduction, and winding.
  void validate() const;
};

// Samples an exact canonical ideal-boson skeleton. time_links_per_beta is output resolution.
[[nodiscard]] IdealBosonConfiguration
sample_ideal_boson_configuration(const Model &model, std::size_t time_links_per_beta,
                                 Random &random,
                                 const NumericalOptions &options = NumericalOptions{});

} // namespace qmc

#endif
