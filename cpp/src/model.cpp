#include "qmc/model.hpp"

#include "qmc/torus_layout.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmc {

namespace {

std::size_t validated_volume(const ModelParameters &parameters) {
  if (!std::isfinite(parameters.beta) || parameters.beta < 0.0) {
    throw std::invalid_argument("beta must be finite and nonnegative");
  }
  if (!std::isfinite(parameters.hopping) || parameters.hopping < 0.0) {
    throw std::invalid_argument("hopping must be finite and nonnegative");
  }
  if (parameters.particle_count > std::numeric_limits<ParticleId>::max()) {
    throw std::invalid_argument("particle_count exceeds the ParticleId range");
  }
  return TorusLayout::checked_volume(parameters.linear_size, parameters.dimension);
}

} // namespace

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

Model::Model(const ModelParameters parameters)
    : particle_count_(parameters.particle_count), beta_(parameters.beta),
      linear_size_(parameters.linear_size), dimension_(parameters.dimension),
      hopping_(parameters.hopping), volume_(validated_volume(parameters)) {}

} // namespace qmc
