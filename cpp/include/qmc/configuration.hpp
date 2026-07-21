#ifndef QMC_CONFIGURATION_HPP
#define QMC_CONFIGURATION_HPP

#include "qmc/free_boson.hpp"
#include "qmc/free_numerics.hpp"
#include "qmc/model.hpp"
#include "qmc/permutation.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qmc {

// Shape-safe row-major covering-space storage in particle, time, axis order.
// Its extent cannot be changed independently of its shape.
class DenseWorldlines {
public:
  DenseWorldlines(std::size_t particle_count, std::size_t point_count, std::size_t dimensions);
  DenseWorldlines(std::size_t particle_count, std::size_t point_count, std::size_t dimensions,
                  std::vector<Coord> values);

  [[nodiscard]] std::size_t particle_count() const noexcept { return particle_count_; }
  [[nodiscard]] std::size_t time_points() const noexcept { return time_points_; }
  [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] std::span<const Coord> values() const noexcept { return values_; }

  // Bounds-checked row-major access in particle, time, axis order.
  [[nodiscard]] Coord &at(ParticleId particle, std::size_t time_index, std::size_t axis);
  [[nodiscard]] const Coord &at(ParticleId particle, std::size_t time_index,
                                std::size_t axis) const;
  [[nodiscard]] std::span<const Coord> site(ParticleId particle, std::size_t time_index) const;

  bool operator==(const DenseWorldlines &) const = default;

private:
  [[nodiscard]] std::size_t site_offset(ParticleId particle, std::size_t time_index) const;

  std::size_t particle_count_;
  std::size_t time_points_;
  std::size_t dimension_;
  std::vector<Coord> values_;
};

// Exact retained-grid ideal-boson state. The permutation is authoritative for
// topology and the covering-space worldlines are authoritative for geometry.
// Torus coordinates and cycle paths/windings are derived rather than stored.
class IdealBosonConfiguration {
public:
  IdealBosonConfiguration(Model model, std::size_t time_links_per_beta, Permutation topology,
                          DenseWorldlines covering_worldlines);

  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] std::size_t time_links_per_beta() const noexcept { return time_links_per_beta_; }
  [[nodiscard]] const Permutation &topology() const noexcept { return topology_; }
  [[nodiscard]] const DenseWorldlines &covering_worldlines() const noexcept {
    return covering_worldlines_;
  }

  // Derives one cycle's covering-space winding from its retained endpoints.
  [[nodiscard]] Site cycle_winding(std::size_t cycle_index) const;

  // Expensive diagnostic audit of shape, topology, endpoint joining, and winding.
  void validate() const;

  bool operator==(const IdealBosonConfiguration &) const = default;

private:
  Model model_;
  std::size_t time_links_per_beta_;
  Permutation topology_;
  DenseWorldlines covering_worldlines_;
};

// Samples an exact canonical ideal-boson skeleton. time_links_per_beta is output resolution.
[[nodiscard]] IdealBosonConfiguration
sample_ideal_boson_configuration(const CanonicalEnsemble &ensemble, std::size_t time_links_per_beta,
                                 Random &random);

// Compatibility overload for a one-off numerical policy. Repeated workflows
// should bind that policy when constructing CanonicalEnsemble.
[[nodiscard]] IdealBosonConfiguration
sample_ideal_boson_configuration(const CanonicalEnsemble &ensemble, std::size_t time_links_per_beta,
                                 Random &random, const NumericalOptions &options);

// One-off convenience overload; repeated workflows should retain a CanonicalEnsemble.
[[nodiscard]] IdealBosonConfiguration
sample_ideal_boson_configuration(const Model &model, std::size_t time_links_per_beta,
                                 Random &random,
                                 const NumericalOptions &options = NumericalOptions{});

} // namespace qmc

#endif
