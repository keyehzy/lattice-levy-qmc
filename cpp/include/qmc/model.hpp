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

// Public construction parameters for an ideal canonical system on an L^d torus.
// Model validates and owns a copy of these values.
struct ModelParameters {
  std::size_t particle_count = 0;
  double beta = 0.0;
  Coord linear_size = 1;
  std::size_t dimension = 1;
  double hopping = 0.0;
};

// Validated immutable physical parameters for an ideal canonical system on an
// L^d torus. Construction rejects invalid scalars, labels that ParticleId
// cannot represent, and lattice volumes that size_t cannot represent.
class Model {
public:
  explicit Model(ModelParameters parameters);

  [[nodiscard]] std::size_t particle_count() const noexcept { return particle_count_; }
  [[nodiscard]] double beta() const noexcept { return beta_; }
  [[nodiscard]] Coord linear_size() const noexcept { return linear_size_; }
  [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] double hopping() const noexcept { return hopping_; }
  [[nodiscard]] std::size_t volume() const noexcept { return volume_; }

  bool operator==(const Model &) const = default;

private:
  std::size_t particle_count_;
  double beta_;
  Coord linear_size_;
  std::size_t dimension_;
  double hopping_;
  std::size_t volume_;
};

} // namespace qmc

#endif
