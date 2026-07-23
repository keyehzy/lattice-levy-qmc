#ifndef QMC_IMAGINARY_TIME_LAGS_HPP
#define QMC_IMAGINARY_TIME_LAGS_HPP

#include "qmc/torus_layout.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qmc {

struct ImaginaryTimeLagRequest {
  // Integer components k_alpha in q_alpha = 2*pi*k_alpha/L.
  std::vector<std::vector<std::size_t>> momentum_indices;
  // Selected evaluation points in the canonical interval [0, beta).
  std::vector<double> lags;
};

// Immutable selected imaginary-time and Fourier geometry. Values that use this
// geometry are ordered first by lag and then by momentum request order.
class ImaginaryTimeLagSet {
public:
  // Requires positive finite beta, nonempty unique momenta and lags, canonical
  // momentum components in [0, L), and finite lags in [0, beta).
  ImaginaryTimeLagSet(double beta, TorusLayout layout, ImaginaryTimeLagRequest request);

  [[nodiscard]] double beta() const noexcept { return beta_; }
  [[nodiscard]] const TorusLayout &layout() const noexcept { return layout_; }
  [[nodiscard]] std::size_t momentum_count() const noexcept;
  [[nodiscard]] std::size_t lag_count() const noexcept;
  [[nodiscard]] std::size_t value_count() const noexcept;
  [[nodiscard]] std::span<const std::size_t> momentum_indices(std::size_t momentum) const;
  [[nodiscard]] double wavevector_component(std::size_t momentum, std::size_t axis) const;
  [[nodiscard]] double lag(std::size_t lag) const;

  bool operator==(const ImaginaryTimeLagSet &) const = default;

private:
  double beta_;
  TorusLayout layout_;
  std::vector<std::size_t> momentum_indices_;
  std::vector<double> lags_;
};

} // namespace qmc

#endif
