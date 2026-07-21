#ifndef QMC_LATTICE_TRANSFORM_DETAIL_HPP
#define QMC_LATTICE_TRANSFORM_DETAIL_HPP

#include "qmc/torus_layout.hpp"

#include <cstddef>
#include <numbers>
#include <span>

namespace qmc::detail {

inline double phase_for_indices(const std::span<const std::size_t> left,
                                const std::span<const std::size_t> right,
                                const TorusLayout &layout) {
  double phase = 0.0;
  for (std::size_t axis = 0; axis < layout.dimension(); ++axis) {
    phase += 2.0 * std::numbers::pi * static_cast<double>(left[axis]) *
             static_cast<double>(right[axis]) / static_cast<double>(layout.linear_size());
  }
  return phase;
}

} // namespace qmc::detail

#endif
