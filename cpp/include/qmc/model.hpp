#ifndef QMC_MODEL_HPP
#define QMC_MODEL_HPP

#include <cstddef>
#include <cstdint>

namespace qmc {

using Coord = std::int64_t;
using ParticleId = std::uint32_t;

// Accuracy and hard-work limits shared by the truncated discrete samplers.
struct NumericalOptions {
  double tail_tolerance = 1e-14;
  std::size_t max_bessel_terms = 2'000'000;
  std::size_t max_winding = 1'000'000;

  void validate() const;
};

// Immutable physical parameters for an ideal canonical system on an L^d torus.
struct Model {
  std::size_t particle_count = 0;
  double beta = 0.0;
  Coord linear_size = 1;
  std::size_t dimension = 1;
  double hopping = 0.0;

  // Validates finite/nonnegative beta and hopping, positive L and d, and label capacity.
  void validate() const;
  // Returns L^d, throwing if it cannot be represented by size_t.
  [[nodiscard]] std::size_t volume() const;
};

} // namespace qmc

#endif
