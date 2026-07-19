#include "qmc/model.hpp"

#include "qmc/torus_layout.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmc {

void NumericalOptions::validate() const {
  if (!std::isfinite(tail_tolerance) || tail_tolerance <= 0.0 || tail_tolerance >= 1.0) {
    throw std::invalid_argument("tail_tolerance must be finite and lie in (0, 1)");
  }
  if (max_bessel_terms == 0) {
    throw std::invalid_argument("max_bessel_terms must be positive");
  }
  if (max_winding == 0) {
    throw std::invalid_argument("max_winding must be positive");
  }
}

void Model::validate() const {
  if (!std::isfinite(beta) || beta < 0.0) {
    throw std::invalid_argument("beta must be finite and nonnegative");
  }
  if (linear_size < 1) {
    throw std::invalid_argument("linear_size must be positive");
  }
  if (dimension < 1) {
    throw std::invalid_argument("dimension must be positive");
  }
  if (!std::isfinite(hopping) || hopping < 0.0) {
    throw std::invalid_argument("hopping must be finite and nonnegative");
  }
  if (particle_count > std::numeric_limits<ParticleId>::max()) {
    throw std::invalid_argument("particle_count exceeds the ParticleId range");
  }
}

std::size_t Model::volume() const {
  validate();
  return TorusLayout::checked_volume(linear_size, dimension);
}

} // namespace qmc
