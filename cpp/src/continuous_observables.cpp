#include "qmc/continuous_observables.hpp"

#include "checked_math.hpp"
#include "continuous_event_sweep.hpp"
#include "continuous_matsubara_detail.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace qmc {
namespace {

std::uint64_t frequency_magnitude(const std::int64_t index) noexcept {
  const auto bits = static_cast<std::uint64_t>(index);
  return index >= 0 ? bits : std::uint64_t{0} - bits;
}

std::size_t add_modulo(const std::size_t first, const std::size_t second,
                       const std::size_t modulus) noexcept {
  return first >= modulus - second ? first - (modulus - second) : first + second;
}

std::size_t multiply_modulo(std::size_t multiplicand, std::size_t multiplier,
                            const std::size_t modulus) noexcept {
  std::size_t result = 0;
  while (multiplier != 0) {
    if ((multiplier & 1U) != 0) {
      result = add_modulo(result, multiplicand, modulus);
    }
    multiplier >>= 1U;
    if (multiplier != 0) {
      multiplicand = add_modulo(multiplicand, multiplicand, modulus);
    }
  }
  return result;
}

void validate_interval(const double begin, const double end, const double beta) {
  if (!std::isfinite(beta) || beta <= 0.0) {
    throw std::invalid_argument("continuous Matsubara beta must be positive and finite");
  }
  if (!std::isfinite(begin) || !std::isfinite(end) || begin < 0.0 || end < begin || end > beta) {
    throw std::invalid_argument("continuous Matsubara interval must lie in [0, beta]");
  }
}

} // namespace

ContinuousMatsubaraPlan::ContinuousMatsubaraPlan(MatsubaraModeSet modes)
    : modes_(std::move(modes)) {
  for (std::size_t frequency = 0; frequency < modes_.frequency_count(); ++frequency) {
    if (frequency_magnitude(modes_.frequency_index(frequency)) >
        static_cast<std::uint64_t>(kMaximumAbsoluteFrequencyIndex)) {
      throw std::invalid_argument(
          "continuous Matsubara frequency exceeds the supported phase-reduction bound");
    }
  }

  const std::size_t component_count =
      detail::checked_product(modes_.momentum_count(), modes_.layout().dimension(),
                              "continuous Matsubara phase extent exceeds size_t");
  if (component_count > site_step_phases_.max_size()) {
    throw std::length_error("continuous Matsubara phase data exceed vector capacity");
  }
  site_step_phases_.reserve(component_count);
  positive_midpoint_phases_.reserve(component_count);
  negative_midpoint_phases_.reserve(component_count);
  for (std::size_t momentum = 0; momentum < modes_.momentum_count(); ++momentum) {
    for (std::size_t axis = 0; axis < modes_.layout().dimension(); ++axis) {
      const double wavevector = modes_.wavevector_component(momentum, axis);
      site_step_phases_.push_back(std::polar(1.0, -wavevector));
      positive_midpoint_phases_.push_back(std::polar(1.0, -0.5 * wavevector));
      negative_midpoint_phases_.push_back(std::polar(1.0, 0.5 * wavevector));
    }
  }
}

ContinuousMeasurementContext::ContinuousMeasurementContext(
    const ContinuousConfiguration &configuration)
    : model_(configuration.model()), layout_(model_.linear_size(), model_.dimension()) {
  detail::ContinuousEventSweepData data =
      detail::build_continuous_event_sweep(configuration, layout_);
  seam_positions_ = std::move(data.seam_positions);
  hops_ = std::move(data.hops);
  event_group_offsets_ = std::move(data.group_offsets);
}

double ContinuousMeasurementContext::event_time(const std::size_t group) const {
  if (group >= event_group_count()) {
    throw std::out_of_range("continuous event group index is out of range");
  }
  return hops_[event_group_offsets_[group]].time;
}

std::span<const ContinuousHop>
ContinuousMeasurementContext::hops_at(const std::size_t group) const {
  if (group >= event_group_count()) {
    throw std::out_of_range("continuous event group index is out of range");
  }
  const std::size_t begin = event_group_offsets_[group];
  const std::size_t end = event_group_offsets_[group + 1];
  return std::span<const ContinuousHop>(hops_).subspan(begin, end - begin);
}

namespace detail {

std::complex<double> matsubara_time_phase(const std::int64_t index, const double time,
                                          const double beta) {
  validate_interval(time, time, beta);
  if (index == 0 || time == 0.0 || time == beta) {
    return {1.0, 0.0};
  }
  const double cycles = static_cast<double>(index) * (time / beta);
  const double reduced_cycles = std::remainder(cycles, 1.0);
  if (reduced_cycles == 0.0) {
    return {1.0, 0.0};
  }
  const double angle = 2.0 * std::numbers::pi * reduced_cycles;
  return {std::cos(angle), std::sin(angle)};
}

std::complex<double> matsubara_interval_transform(const std::int64_t index, const double begin,
                                                  const double end, const double beta) {
  validate_interval(begin, end, beta);
  const double duration = end - begin;
  if (duration == 0.0) {
    return {0.0, 0.0};
  }
  if (index == 0) {
    return {duration, 0.0};
  }
  if (begin == 0.0 && end == beta) {
    return {0.0, 0.0};
  }

  const double cycles = static_cast<double>(index) * (duration / beta);
  const double reduced_cycles = std::remainder(cycles, 2.0);
  double sinc = 0.0;
  if (cycles == 0.0) {
    sinc = 1.0;
  } else if (reduced_cycles != 0.0 && std::abs(reduced_cycles) != 1.0) {
    const double argument = std::numbers::pi * cycles;
    if (std::abs(argument) < 1.0e-4) {
      const double square = argument * argument;
      sinc = 1.0 - (square / 6.0) + (square * square / 120.0);
    } else {
      sinc = std::sin(std::numbers::pi * reduced_cycles) / argument;
    }
  }
  const double midpoint = begin + (duration / 2.0);
  return duration * sinc * matsubara_time_phase(index, midpoint, beta);
}

std::complex<double> matsubara_site_phase(const MatsubaraModeSet &modes, const std::size_t momentum,
                                          const SiteId site) {
  const auto momentum_components = modes.momentum_indices(momentum);
  const std::vector<std::size_t> site_components = modes.layout().decode(site);
  const auto linear_size = static_cast<std::size_t>(modes.layout().linear_size());
  std::complex<double> phase{1.0, 0.0};
  for (std::size_t axis = 0; axis < modes.layout().dimension(); ++axis) {
    const std::size_t residue =
        multiply_modulo(momentum_components[axis], site_components[axis], linear_size);
    const double angle =
        -2.0 * std::numbers::pi * static_cast<double>(residue) / static_cast<double>(linear_size);
    phase *= std::polar(1.0, angle);
  }
  return phase;
}

} // namespace detail

} // namespace qmc
