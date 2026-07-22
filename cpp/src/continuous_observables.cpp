#include "qmc/continuous_observables.hpp"

#include "checked_math.hpp"
#include "continuous_event_sweep.hpp"
#include "continuous_matsubara_detail.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

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

class CompensatedComplexSum {
public:
  void add(const std::complex<double> value) noexcept {
    add_component(value.real(), real_sum_, real_correction_);
    add_component(value.imag(), imaginary_sum_, imaginary_correction_);
  }

  [[nodiscard]] std::complex<double> value() const noexcept {
    return {real_sum_ + real_correction_, imaginary_sum_ + imaginary_correction_};
  }

private:
  static void add_component(const double value, double &sum, double &correction) noexcept {
    const double next = sum + value;
    if (std::abs(sum) >= std::abs(value)) {
      correction += (sum - next) + value;
    } else {
      correction += (value - next) + sum;
    }
    sum = next;
  }

  double real_sum_ = 0.0;
  double real_correction_ = 0.0;
  double imaginary_sum_ = 0.0;
  double imaginary_correction_ = 0.0;
};

template <class T>
std::vector<T> make_checked_vector(const std::size_t extent, const char *description) {
  if (extent > std::vector<T>{}.max_size()) {
    throw std::length_error(description);
  }
  return std::vector<T>(extent);
}

bool is_finite(const std::complex<double> value) noexcept {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

bool is_zero_momentum(const MatsubaraModeSet &modes, const std::size_t momentum) {
  const auto components = modes.momentum_indices(momentum);
  return std::ranges::all_of(components,
                             [](const std::size_t component) { return component == 0; });
}

void validate_projection_geometry(const ContinuousMeasurementContext &context,
                                  const MatsubaraModeSet &modes) {
  if (context.model().beta() != modes.beta() || context.layout() != modes.layout()) {
    throw std::invalid_argument(
        "continuous measurement context and Matsubara plan geometry differ");
  }
}

struct ProjectedParticleModeData {
  std::vector<std::complex<double>> density_values;
  std::vector<std::complex<double>> flux_values;
  std::vector<std::size_t> axis_event_counts;
};

class ContinuousParticleProjector {
public:
  ContinuousParticleProjector(const ContinuousMeasurementContext &context,
                              const MatsubaraModeSet &modes,
                              const std::span<const std::complex<double>> site_step_phases,
                              const std::span<const std::complex<double>> positive_midpoint_phases,
                              const std::span<const std::complex<double>> negative_midpoint_phases)
      : context_(context), modes_(modes), site_step_phases_(site_step_phases),
        positive_midpoint_phases_(positive_midpoint_phases),
        negative_midpoint_phases_(negative_midpoint_phases),
        particle_count_(context.model().particle_count()), momentum_count_(modes.momentum_count()),
        frequency_count_(modes.frequency_count()), dimension_(context.model().dimension()),
        mode_count_(modes.mode_count()),
        flux_extent_(detail::checked_product(
            mode_count_, dimension_, "continuous particle-mode flux extent exceeds size_t")),
        particle_phase_extent_(detail::checked_product(
            particle_count_, momentum_count_, "continuous particle phase extent exceeds size_t")),
        density_amplitudes_(make_checked_vector<CompensatedComplexSum>(
            mode_count_, "continuous density extent exceeds vector capacity")),
        flux_amplitudes_(make_checked_vector<CompensatedComplexSum>(
            flux_extent_, "continuous flux extent exceeds vector capacity")),
        current_density_(make_checked_vector<CompensatedComplexSum>(
            momentum_count_, "continuous density-state extent exceeds vector capacity")),
        particle_phases_(make_checked_vector<std::complex<double>>(
            particle_phase_extent_, "continuous particle phase extent exceeds vector capacity")),
        time_phases_(make_checked_vector<std::complex<double>>(
            frequency_count_, "continuous time-phase extent exceeds vector capacity")),
        axis_event_counts_(make_checked_vector<std::size_t>(
            dimension_, "continuous axis-count extent exceeds vector capacity")),
        signed_axis_displacements_(make_checked_vector<Coord>(
            dimension_, "continuous displacement extent exceeds vector capacity")),
        zero_momenta_(make_checked_vector<std::uint8_t>(
            momentum_count_, "continuous zero-momentum extent exceeds vector capacity")) {
    validate_phase_extents();
    initialize_zero_momenta();
    initialize_particle_phases();
  }

  [[nodiscard]] ProjectedParticleModeData project() {
    double previous_time = 0.0;
    for (std::size_t group = 0; group < context_.event_group_count(); ++group) {
      const double time = context_.event_time(group);
      add_residence_interval(previous_time, time);
      load_time_phases(time);
      apply_event_group(group);
      previous_time = time;
    }
    add_residence_interval(previous_time, modes_.beta());
    return finish();
  }

private:
  void validate_phase_extents() const {
    const std::size_t component_extent = detail::checked_product(
        momentum_count_, dimension_, "continuous Matsubara component extent exceeds size_t");
    if (site_step_phases_.size() != component_extent ||
        positive_midpoint_phases_.size() != component_extent ||
        negative_midpoint_phases_.size() != component_extent) {
      throw std::logic_error("continuous Matsubara plan has inconsistent phase storage");
    }
  }

  void initialize_zero_momenta() {
    for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
      zero_momenta_[momentum] = is_zero_momentum(modes_, momentum) ? 1U : 0U;
    }
  }

  void initialize_particle_phases() {
    for (std::size_t particle = 0; particle < particle_count_; ++particle) {
      for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
        const std::complex<double> phase =
            detail::matsubara_site_phase(modes_, momentum, context_.seam_positions()[particle]);
        particle_phases_[particle_phase_index(particle, momentum)] = phase;
        current_density_[momentum].add(phase);
      }
    }
  }

  void add_residence_interval(const double begin, const double end) {
    for (std::size_t frequency = 0; frequency < frequency_count_; ++frequency) {
      const std::complex<double> interval = detail::matsubara_interval_transform(
          modes_.frequency_index(frequency), begin, end, modes_.beta());
      for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
        density_amplitudes_[mode_index(frequency, momentum)].add(
            current_density_[momentum].value() * interval);
      }
    }
  }

  void load_time_phases(const double time) {
    for (std::size_t frequency = 0; frequency < frequency_count_; ++frequency) {
      time_phases_[frequency] =
          detail::matsubara_time_phase(modes_.frequency_index(frequency), time, modes_.beta());
    }
  }

  void apply_event_group(const std::size_t group) {
    for (const ContinuousHop &hop : context_.hops_at(group)) {
      apply_hop(hop);
    }
  }

  void apply_hop(const ContinuousHop &hop) {
    const auto axis = static_cast<std::size_t>(hop.axis);
    ++axis_event_counts_[axis];
    signed_axis_displacements_[axis] =
        detail::checked_add(signed_axis_displacements_[axis], static_cast<Coord>(hop.direction),
                            "continuous signed event displacement exceeds int64 range");
    for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
      apply_hop_momentum(hop, momentum, axis);
    }
  }

  void apply_hop_momentum(const ContinuousHop &hop, const std::size_t momentum,
                          const std::size_t axis) {
    const std::size_t particle_phase =
        particle_phase_index(static_cast<std::size_t>(hop.particle), momentum);
    const std::size_t component = component_index(momentum, axis);
    const std::complex<double> departure_phase = particle_phases_[particle_phase];
    const bool positive = hop.direction > 0;
    const std::complex<double> midpoint_phase =
        positive ? positive_midpoint_phases_[component] : negative_midpoint_phases_[component];
    const std::complex<double> impulse =
        static_cast<double>(hop.direction) * departure_phase * midpoint_phase;
    add_flux_impulse(momentum, axis, impulse);

    const std::complex<double> step_phase = site_step_phases_[component];
    const std::complex<double> arrival_phase =
        departure_phase * (positive ? step_phase : std::conj(step_phase));
    current_density_[momentum].add(arrival_phase - departure_phase);
    particle_phases_[particle_phase] = arrival_phase;
  }

  void add_flux_impulse(const std::size_t momentum, const std::size_t axis,
                        const std::complex<double> impulse) {
    for (std::size_t frequency = 0; frequency < frequency_count_; ++frequency) {
      const std::size_t mode = mode_index(frequency, momentum);
      flux_amplitudes_[flux_index(mode, axis)].add(impulse * time_phases_[frequency]);
    }
  }

  [[nodiscard]] ProjectedParticleModeData finish() const {
    auto density_values = make_checked_vector<std::complex<double>>(
        mode_count_, "continuous density result extent exceeds vector capacity");
    auto flux_values = make_checked_vector<std::complex<double>>(
        flux_extent_, "continuous flux result extent exceeds vector capacity");
    for (std::size_t frequency = 0; frequency < frequency_count_; ++frequency) {
      for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
        const std::size_t mode = mode_index(frequency, momentum);
        density_values[mode] = finished_density(frequency, momentum, mode);
        for (std::size_t axis = 0; axis < dimension_; ++axis) {
          flux_values[flux_index(mode, axis)] = finished_flux(frequency, momentum, mode, axis);
        }
      }
    }
    return {
        .density_values = std::move(density_values),
        .flux_values = std::move(flux_values),
        .axis_event_counts = axis_event_counts_,
    };
  }

  [[nodiscard]] std::complex<double> finished_density(const std::size_t frequency,
                                                      const std::size_t momentum,
                                                      const std::size_t mode) const {
    std::complex<double> value = density_amplitudes_[mode].value();
    if (zero_momenta_[momentum] != 0U) {
      value = {0.0, 0.0};
      if (modes_.frequency_index(frequency) == 0) {
        value = {modes_.beta() * static_cast<double>(particle_count_), 0.0};
      }
    }
    if (!is_finite(value)) {
      throw std::overflow_error("continuous density amplitude is non-finite");
    }
    return value;
  }

  [[nodiscard]] std::complex<double> finished_flux(const std::size_t frequency,
                                                   const std::size_t momentum,
                                                   const std::size_t mode,
                                                   const std::size_t axis) const {
    std::complex<double> value = flux_amplitudes_[flux_index(mode, axis)].value();
    if (zero_momenta_[momentum] != 0U && modes_.frequency_index(frequency) == 0) {
      value = {static_cast<double>(signed_axis_displacements_[axis]), 0.0};
    }
    if (!is_finite(value)) {
      throw std::overflow_error("continuous hopping-flux amplitude is non-finite");
    }
    return value;
  }

  [[nodiscard]] std::size_t mode_index(const std::size_t frequency,
                                       const std::size_t momentum) const noexcept {
    return (frequency * momentum_count_) + momentum;
  }

  [[nodiscard]] std::size_t flux_index(const std::size_t mode,
                                       const std::size_t axis) const noexcept {
    return (mode * dimension_) + axis;
  }

  [[nodiscard]] std::size_t component_index(const std::size_t momentum,
                                            const std::size_t axis) const noexcept {
    return (momentum * dimension_) + axis;
  }

  [[nodiscard]] std::size_t particle_phase_index(const std::size_t particle,
                                                 const std::size_t momentum) const noexcept {
    return (particle * momentum_count_) + momentum;
  }

  const ContinuousMeasurementContext &context_;
  const MatsubaraModeSet &modes_;
  std::span<const std::complex<double>> site_step_phases_;
  std::span<const std::complex<double>> positive_midpoint_phases_;
  std::span<const std::complex<double>> negative_midpoint_phases_;
  std::size_t particle_count_;
  std::size_t momentum_count_;
  std::size_t frequency_count_;
  std::size_t dimension_;
  std::size_t mode_count_;
  std::size_t flux_extent_;
  std::size_t particle_phase_extent_;
  std::vector<CompensatedComplexSum> density_amplitudes_;
  std::vector<CompensatedComplexSum> flux_amplitudes_;
  std::vector<CompensatedComplexSum> current_density_;
  std::vector<std::complex<double>> particle_phases_;
  std::vector<std::complex<double>> time_phases_;
  std::vector<std::size_t> axis_event_counts_;
  std::vector<Coord> signed_axis_displacements_;
  std::vector<std::uint8_t> zero_momenta_;
};

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
      const std::complex<double> positive_midpoint = std::polar(1.0, -0.5 * wavevector);
      site_step_phases_.push_back(std::polar(1.0, -wavevector));
      positive_midpoint_phases_.push_back(positive_midpoint);
      negative_midpoint_phases_.push_back(std::conj(positive_midpoint));
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

ContinuousParticleModes::ContinuousParticleModes(Model model, MatsubaraModeSet modes,
                                                 std::vector<std::complex<double>> density_values,
                                                 std::vector<std::complex<double>> flux_values,
                                                 std::vector<std::size_t> axis_event_counts)
    : model_(model), density_(std::move(modes), std::move(density_values)),
      flux_values_(std::move(flux_values)), axis_event_counts_(std::move(axis_event_counts)) {
  if (model_.beta() != density_.modes().beta() ||
      TorusLayout(model_.linear_size(), model_.dimension()) != density_.modes().layout()) {
    throw std::invalid_argument("continuous particle-mode geometry does not match its model");
  }
  const std::size_t flux_extent = detail::checked_product(
      density_.modes().mode_count(), model_.dimension(), "continuous flux extent exceeds size_t");
  if (flux_values_.size() != flux_extent) {
    throw std::invalid_argument("continuous particle modes have the wrong flux extent");
  }
  if (axis_event_counts_.size() != model_.dimension()) {
    throw std::invalid_argument("continuous particle modes have the wrong axis-count extent");
  }
}

std::complex<double> ContinuousParticleModes::density(const std::size_t frequency,
                                                      const std::size_t momentum) const {
  return density_.at(frequency, momentum);
}

std::complex<double> ContinuousParticleModes::flux(const std::size_t frequency,
                                                   const std::size_t momentum,
                                                   const std::size_t axis) const {
  if (frequency >= modes().frequency_count()) {
    throw std::out_of_range("continuous flux frequency index is out of range");
  }
  if (momentum >= modes().momentum_count()) {
    throw std::out_of_range("continuous flux momentum index is out of range");
  }
  if (axis >= model_.dimension()) {
    throw std::out_of_range("continuous flux axis is out of range");
  }
  const std::size_t mode = (frequency * modes().momentum_count()) + momentum;
  return flux_values_[(mode * model_.dimension()) + axis];
}

std::size_t ContinuousParticleModes::axis_event_count(const std::size_t axis) const {
  if (axis >= model_.dimension()) {
    throw std::out_of_range("continuous event-count axis is out of range");
  }
  return axis_event_counts_[axis];
}

ContinuousParticleModes continuous_particle_modes(const ContinuousMeasurementContext &context,
                                                  const ContinuousMatsubaraPlan &plan) {
  const MatsubaraModeSet &modes = plan.modes();
  validate_projection_geometry(context, modes);
  ContinuousParticleProjector projector(context, modes, plan.site_step_phases_,
                                        plan.positive_midpoint_phases_,
                                        plan.negative_midpoint_phases_);
  ProjectedParticleModeData projected = projector.project();
  return {context.model(), modes, std::move(projected.density_values),
          std::move(projected.flux_values), std::move(projected.axis_event_counts)};
}

ContinuousParticleModes continuous_particle_modes(const ContinuousConfiguration &configuration,
                                                  const ContinuousMatsubaraPlan &plan) {
  return continuous_particle_modes(ContinuousMeasurementContext(configuration), plan);
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
