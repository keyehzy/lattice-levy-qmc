#ifndef QMC_MATSUBARA_MODES_HPP
#define QMC_MATSUBARA_MODES_HPP

#include "qmc/torus_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc {

struct MatsubaraModeRequest {
  // Integer components k_alpha in q_alpha = 2*pi*k_alpha/L.
  std::vector<std::vector<std::size_t>> momentum_indices;
  // Signed n in omega_n = 2*pi*n/beta.
  std::vector<std::int64_t> frequency_indices;
};

// Immutable selected Fourier geometry. Modes use frequency-major ordering,
// followed by momentum request order.
class MatsubaraModeSet {
public:
  // Requires positive finite beta, nonempty unique momenta and frequencies,
  // and canonical momentum components in [0, L). Checked shape arithmetic and
  // unrepresentable physical frequencies throw overflow_error.
  MatsubaraModeSet(double beta, TorusLayout layout, MatsubaraModeRequest request);

  [[nodiscard]] double beta() const noexcept { return beta_; }
  [[nodiscard]] const TorusLayout &layout() const noexcept { return layout_; }
  [[nodiscard]] std::size_t momentum_count() const noexcept;
  [[nodiscard]] std::size_t frequency_count() const noexcept;
  [[nodiscard]] std::size_t mode_count() const noexcept;
  [[nodiscard]] std::span<const std::size_t> momentum_indices(std::size_t momentum) const;
  [[nodiscard]] double wavevector_component(std::size_t momentum, std::size_t axis) const;
  [[nodiscard]] std::int64_t frequency_index(std::size_t frequency) const;
  [[nodiscard]] double frequency(std::size_t frequency) const;

  bool operator==(const MatsubaraModeSet &) const = default;

private:
  double beta_;
  TorusLayout layout_;
  std::vector<std::size_t> momentum_indices_;
  std::vector<std::int64_t> frequency_indices_;
};

// Shape-safe scalar field over one selected Matsubara mode set.
template <class T> class MatsubaraModeField {
public:
  // Requires exactly modes.mode_count() values.
  MatsubaraModeField(MatsubaraModeSet modes, std::vector<T> values)
      : modes_(std::move(modes)), values_(std::move(values)) {
    if (values_.size() != modes_.mode_count()) {
      throw std::invalid_argument("Matsubara mode field has the wrong value extent");
    }
  }

  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept { return modes_; }
  [[nodiscard]] std::span<const T> values() const noexcept { return values_; }
  [[nodiscard]] const T &at(const std::size_t frequency, const std::size_t momentum) const {
    if (frequency >= modes_.frequency_count()) {
      throw std::out_of_range("Matsubara frequency index is out of range");
    }
    if (momentum >= modes_.momentum_count()) {
      throw std::out_of_range("Matsubara momentum index is out of range");
    }
    return values_[(frequency * modes_.momentum_count()) + momentum];
  }

  bool operator==(const MatsubaraModeField &) const = default;

private:
  MatsubaraModeSet modes_;
  std::vector<T> values_;
};

} // namespace qmc

#endif
