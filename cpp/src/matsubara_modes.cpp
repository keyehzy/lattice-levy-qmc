#include "qmc/matsubara_modes.hpp"

#include "checked_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace qmc {
namespace {

double physical_frequency(const std::int64_t index, const double beta) {
  const double value = (2.0 * std::numbers::pi * static_cast<double>(index)) / beta;
  if (!std::isfinite(value) || (index != 0 && value == 0.0)) {
    throw std::overflow_error("Matsubara frequency is not representable as double");
  }
  return value;
}

} // namespace

MatsubaraModeSet::MatsubaraModeSet(double beta, TorusLayout layout, MatsubaraModeRequest request)
    : beta_(beta), layout_(std::move(layout)) {
  if (!std::isfinite(beta_) || beta_ <= 0.0) {
    throw std::invalid_argument("Matsubara beta must be positive and finite");
  }
  if (request.momentum_indices.empty()) {
    throw std::invalid_argument("Matsubara mode request requires at least one momentum");
  }
  if (request.frequency_indices.empty()) {
    throw std::invalid_argument("Matsubara mode request requires at least one frequency");
  }

  const std::size_t component_count =
      detail::checked_product(request.momentum_indices.size(), layout_.dimension(),
                              "Matsubara momentum component extent exceeds size_t");
  static_cast<void>(detail::checked_product(request.frequency_indices.size(),
                                            request.momentum_indices.size(),
                                            "Matsubara mode extent exceeds size_t"));
  if (component_count > momentum_indices_.max_size()) {
    throw std::length_error("Matsubara momentum components exceed vector capacity");
  }
  if (request.frequency_indices.size() > frequency_indices_.max_size()) {
    throw std::length_error("Matsubara frequencies exceed vector capacity");
  }

  const auto linear_size = static_cast<std::size_t>(layout_.linear_size());
  for (const auto &indices : request.momentum_indices) {
    if (indices.size() != layout_.dimension()) {
      throw std::invalid_argument("Matsubara momentum has the wrong dimension");
    }
    if (std::ranges::any_of(
            indices, [linear_size](const std::size_t index) { return index >= linear_size; })) {
      throw std::invalid_argument("Matsubara momentum component is outside [0, L)");
    }
  }
  std::vector<std::size_t> momentum_order(request.momentum_indices.size());
  std::iota(momentum_order.begin(), momentum_order.end(), std::size_t{0});
  std::ranges::sort(momentum_order, [&request](const std::size_t left, const std::size_t right) {
    return request.momentum_indices[left] < request.momentum_indices[right];
  });
  for (std::size_t index = 1; index < momentum_order.size(); ++index) {
    if (request.momentum_indices[momentum_order[index - 1]] ==
        request.momentum_indices[momentum_order[index]]) {
      throw std::invalid_argument("Matsubara mode request contains a duplicate momentum");
    }
  }

  for (const std::int64_t frequency : request.frequency_indices) {
    static_cast<void>(physical_frequency(frequency, beta_));
  }
  std::vector<std::int64_t> ordered_frequencies = request.frequency_indices;
  std::ranges::sort(ordered_frequencies);
  if (std::ranges::adjacent_find(ordered_frequencies) != ordered_frequencies.end()) {
    throw std::invalid_argument("Matsubara mode request contains a duplicate frequency");
  }

  momentum_indices_.reserve(component_count);
  for (const auto &indices : request.momentum_indices) {
    momentum_indices_.insert(momentum_indices_.end(), indices.begin(), indices.end());
  }
  frequency_indices_ = std::move(request.frequency_indices);
}

std::size_t MatsubaraModeSet::momentum_count() const noexcept {
  return momentum_indices_.size() / layout_.dimension();
}

std::size_t MatsubaraModeSet::frequency_count() const noexcept { return frequency_indices_.size(); }

std::size_t MatsubaraModeSet::mode_count() const noexcept {
  return frequency_count() * momentum_count();
}

std::span<const std::size_t> MatsubaraModeSet::momentum_indices(const std::size_t momentum) const {
  if (momentum >= momentum_count()) {
    throw std::out_of_range("Matsubara momentum index is out of range");
  }
  return std::span<const std::size_t>(momentum_indices_)
      .subspan(momentum * layout_.dimension(), layout_.dimension());
}

double MatsubaraModeSet::wavevector_component(const std::size_t momentum,
                                              const std::size_t axis) const {
  const auto indices = momentum_indices(momentum);
  if (axis >= layout_.dimension()) {
    throw std::out_of_range("Matsubara momentum axis is out of range");
  }
  return 2.0 * std::numbers::pi * static_cast<double>(indices[axis]) /
         static_cast<double>(layout_.linear_size());
}

std::int64_t MatsubaraModeSet::frequency_index(const std::size_t frequency) const {
  if (frequency >= frequency_count()) {
    throw std::out_of_range("Matsubara frequency index is out of range");
  }
  return frequency_indices_[frequency];
}

double MatsubaraModeSet::frequency(const std::size_t frequency) const {
  return physical_frequency(frequency_index(frequency), beta_);
}

} // namespace qmc
