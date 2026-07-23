#include "qmc/imaginary_time_lags.hpp"

#include "checked_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc {

ImaginaryTimeLagSet::ImaginaryTimeLagSet(double beta, TorusLayout layout,
                                         ImaginaryTimeLagRequest request)
    : beta_(beta), layout_(std::move(layout)) {
  if (!std::isfinite(beta_) || beta_ <= 0.0) {
    throw std::invalid_argument("imaginary-time beta must be positive and finite");
  }
  if (request.momentum_indices.empty()) {
    throw std::invalid_argument("imaginary-time lag request requires at least one momentum");
  }
  if (request.lags.empty()) {
    throw std::invalid_argument("imaginary-time lag request requires at least one lag");
  }

  const std::size_t component_count =
      detail::checked_product(request.momentum_indices.size(), layout_.dimension(),
                              "imaginary-time momentum component extent exceeds size_t");
  static_cast<void>(detail::checked_product(request.lags.size(), request.momentum_indices.size(),
                                            "imaginary-time lag-value extent exceeds size_t"));
  if (component_count > momentum_indices_.max_size()) {
    throw std::length_error("imaginary-time momentum components exceed vector capacity");
  }
  if (request.lags.size() > lags_.max_size()) {
    throw std::length_error("imaginary-time lags exceed vector capacity");
  }

  const auto linear_size = static_cast<std::size_t>(layout_.linear_size());
  for (const auto &indices : request.momentum_indices) {
    if (indices.size() != layout_.dimension()) {
      throw std::invalid_argument("imaginary-time momentum has the wrong dimension");
    }
    if (std::ranges::any_of(
            indices, [linear_size](const std::size_t index) { return index >= linear_size; })) {
      throw std::invalid_argument("imaginary-time momentum component is outside [0, L)");
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
      throw std::invalid_argument("imaginary-time lag request contains a duplicate momentum");
    }
  }

  for (double &lag : request.lags) {
    if (!std::isfinite(lag) || lag < 0.0 || lag >= beta_) {
      throw std::invalid_argument("imaginary-time lag must lie in [0, beta)");
    }
    if (lag == 0.0) {
      lag = 0.0;
    }
  }
  std::vector<double> ordered_lags = request.lags;
  std::ranges::sort(ordered_lags);
  if (std::ranges::adjacent_find(ordered_lags) != ordered_lags.end()) {
    throw std::invalid_argument("imaginary-time lag request contains a duplicate lag");
  }

  momentum_indices_.reserve(component_count);
  for (const auto &indices : request.momentum_indices) {
    momentum_indices_.insert(momentum_indices_.end(), indices.begin(), indices.end());
  }
  lags_ = std::move(request.lags);
}

std::size_t ImaginaryTimeLagSet::momentum_count() const noexcept {
  return momentum_indices_.size() / layout_.dimension();
}

std::size_t ImaginaryTimeLagSet::lag_count() const noexcept { return lags_.size(); }

std::size_t ImaginaryTimeLagSet::value_count() const noexcept {
  return lag_count() * momentum_count();
}

std::span<const std::size_t>
ImaginaryTimeLagSet::momentum_indices(const std::size_t momentum) const {
  if (momentum >= momentum_count()) {
    throw std::out_of_range("imaginary-time momentum index is out of range");
  }
  return std::span<const std::size_t>(momentum_indices_)
      .subspan(momentum * layout_.dimension(), layout_.dimension());
}

double ImaginaryTimeLagSet::wavevector_component(const std::size_t momentum,
                                                 const std::size_t axis) const {
  const auto indices = momentum_indices(momentum);
  if (axis >= layout_.dimension()) {
    throw std::out_of_range("imaginary-time momentum axis is out of range");
  }
  return 2.0 * std::numbers::pi * static_cast<double>(indices[axis]) /
         static_cast<double>(layout_.linear_size());
}

double ImaginaryTimeLagSet::lag(const std::size_t lag) const {
  if (lag >= lag_count()) {
    throw std::out_of_range("imaginary-time lag index is out of range");
  }
  return lags_[lag];
}

} // namespace qmc
