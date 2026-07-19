#ifndef QMC_INTERACTING_MODEL_HPP
#define QMC_INTERACTING_MODEL_HPP

#include "qmc/model.hpp"

namespace qmc {

// Canonical Bose-Hubbard parameters. The embedded Model defines the ideal
// reference measure; interaction is the on-site pair coupling U.
struct InteractingModel {
  Model free;
  double interaction = 0.0;

  // Interacting paths require a positive inverse temperature and finite U.
  void validate() const;
  [[nodiscard]] std::size_t volume() const;
};

} // namespace qmc

#endif
