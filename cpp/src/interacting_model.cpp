#include "qmc/interacting_model.hpp"

#include <cmath>
#include <stdexcept>

namespace qmc {

void InteractingModel::validate() const {
  free.validate();
  if (free.beta <= 0.0) {
    throw std::invalid_argument("interacting beta must be finite and positive");
  }
  if (!std::isfinite(interaction)) {
    throw std::invalid_argument("interaction must be finite");
  }
}

std::size_t InteractingModel::volume() const {
  validate();
  return free.volume();
}

} // namespace qmc
