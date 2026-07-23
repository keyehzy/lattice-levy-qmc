#include "qmc/continuous_observables.hpp"

#include "checked_math.hpp"
#include "continuous_event_sweep.hpp"
#include "continuous_matsubara_detail.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <unordered_map>
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

std::complex<double> spatial_site_phase(const TorusLayout &layout,
                                        const std::span<const std::size_t> momentum_components,
                                        const SiteId site) {
  if (momentum_components.size() != layout.dimension()) {
    throw std::logic_error("continuous spatial phase has the wrong momentum dimension");
  }
  const std::vector<std::size_t> site_components = layout.decode(site);
  const auto linear_size = static_cast<std::size_t>(layout.linear_size());
  std::complex<double> phase{1.0, 0.0};
  for (std::size_t axis = 0; axis < layout.dimension(); ++axis) {
    const std::size_t residue =
        multiply_modulo(momentum_components[axis], site_components[axis], linear_size);
    const double angle =
        -2.0 * std::numbers::pi * static_cast<double>(residue) / static_cast<double>(linear_size);
    phase *= std::polar(1.0, angle);
  }
  return phase;
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

class CompensatedRealSum {
public:
  void add(const double value) noexcept {
    const double next = sum_ + value;
    if (std::abs(sum_) >= std::abs(value)) {
      correction_ += (sum_ - next) + value;
    } else {
      correction_ += (value - next) + sum_;
    }
    sum_ = next;
  }

  [[nodiscard]] double value() const noexcept { return sum_ + correction_; }

private:
  double sum_ = 0.0;
  double correction_ = 0.0;
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

void validate_model_mode_geometry(const Model &model, const MatsubaraModeSet &modes,
                                  const char *description) {
  if (model.beta() != modes.beta() ||
      TorusLayout(model.linear_size(), model.dimension()) != modes.layout()) {
    throw std::invalid_argument(description);
  }
}

MatsubaraModeSet validated_density_accumulator_modes(const Model &model, MatsubaraModeSet modes) {
  validate_model_mode_geometry(model, modes,
                               "density accumulator model and Matsubara geometry differ");
  return modes;
}

std::size_t validated_measurements_per_block(const std::size_t measurements_per_block) {
  if (measurements_per_block == 0) {
    throw std::invalid_argument("density block size must be positive");
  }
  return measurements_per_block;
}

std::size_t density_covariance_extent(const MatsubaraModeSet &modes) {
  const std::size_t frequency_square =
      detail::checked_product(modes.frequency_count(), modes.frequency_count(),
                              "density covariance frequency extent exceeds size_t");
  return detail::checked_product(modes.momentum_count(), frequency_square,
                                 "density covariance extent exceeds size_t");
}

std::vector<std::complex<double>> density_analytic_means(const Model &model,
                                                         const MatsubaraModeSet &modes) {
  auto means = make_checked_vector<std::complex<double>>(
      modes.mode_count(), "density analytic means exceed vector capacity");
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    if (modes.frequency_index(frequency) != 0) {
      continue;
    }
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      if (!is_zero_momentum(modes, momentum)) {
        continue;
      }
      const double zero_mode_mean = model.beta() * static_cast<double>(model.particle_count());
      if (!std::isfinite(zero_mode_mean)) {
        throw std::overflow_error("density analytic zero-mode mean is non-finite");
      }
      means[(frequency * modes.momentum_count()) + momentum] = {zero_mode_mean, 0.0};
    }
  }
  return means;
}

struct DensityModeObservation {
  std::vector<std::complex<double>> amplitudes;
  std::vector<double> connected_values;
};

DensityModeObservation
density_mode_observation(const ContinuousParticleModes &values, const Model &model,
                         const MatsubaraModeSet &modes,
                         const std::span<const std::complex<double>> analytic_means) {
  if (values.model() != model) {
    throw std::invalid_argument("density observation has a different model");
  }
  if (values.modes() != modes) {
    throw std::invalid_argument("density observation has different Matsubara modes");
  }
  if (analytic_means.size() != modes.mode_count()) {
    throw std::logic_error("density analytic means have the wrong extent");
  }

  DensityModeObservation observation{
      .amplitudes = make_checked_vector<std::complex<double>>(
          modes.mode_count(), "density observation amplitudes exceed vector capacity"),
      .connected_values = make_checked_vector<double>(
          modes.mode_count(), "density observations exceed vector capacity"),
  };
  const auto volume_divisor = static_cast<double>(model.volume());
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      const std::size_t mode = (frequency * modes.momentum_count()) + momentum;
      const std::complex<double> amplitude = values.density(frequency, momentum);
      if (!is_finite(amplitude)) {
        throw std::overflow_error("density observation amplitude is non-finite");
      }
      observation.amplitudes[mode] = amplitude;

      const std::complex<double> centered = amplitude - analytic_means[mode];
      if (!is_finite(centered)) {
        throw std::overflow_error("centered density amplitude is non-finite");
      }
      const double centered_norm = std::norm(centered);
      if (!std::isfinite(centered_norm)) {
        throw std::overflow_error("density susceptibility observation is non-finite");
      }
      const double connected_value = (centered_norm / model.beta()) / volume_divisor;
      if (!std::isfinite(connected_value) || connected_value < 0.0) {
        throw std::overflow_error("normalized density observation is non-finite");
      }
      observation.connected_values[mode] = connected_value;
    }
  }
  return observation;
}

std::size_t validate_density_block_series(const MatsubaraModeSet &modes,
                                          const std::size_t measurements_per_block,
                                          const std::span<const double> block_values,
                                          const std::size_t sample_count) {
  if (block_values.size() % modes.mode_count() != 0) {
    throw std::invalid_argument("density block series has the wrong block-value extent");
  }
  const std::size_t block_count = block_values.size() / modes.mode_count();
  if (block_count < 2) {
    throw std::invalid_argument("density block series requires at least two blocks");
  }
  const std::size_t expected_sample_count = detail::checked_product(
      block_count, measurements_per_block, "density block-series sample count exceeds size_t");
  if (sample_count != expected_sample_count) {
    throw std::invalid_argument("density block series has an inconsistent sample count");
  }
  for (const double value : block_values) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::invalid_argument("density block value is invalid");
    }
  }
  return block_count;
}

std::vector<double> density_block_means(const MatsubaraModeSet &modes,
                                        const std::span<const double> block_values,
                                        const std::size_t block_count) {
  auto means =
      make_checked_vector<double>(modes.mode_count(), "density block means exceed vector capacity");
  const auto block_divisor = static_cast<double>(block_count);
  for (std::size_t mode = 0; mode < modes.mode_count(); ++mode) {
    CompensatedRealSum mean_sum;
    for (std::size_t block = 0; block < block_count; ++block) {
      mean_sum.add(block_values[(block * modes.mode_count()) + mode] / block_divisor);
    }
    means[mode] = mean_sum.value();
    if (!std::isfinite(means[mode]) || means[mode] < 0.0) {
      throw std::overflow_error("density block-series mean is non-finite");
    }
  }
  return means;
}

double density_covariance_of_mean(const std::span<const double> block_values,
                                  const std::span<const double> means, const std::size_t mode_count,
                                  const std::size_t block_count, const std::size_t left_mode,
                                  const std::size_t right_mode, const bool diagonal) {
  CompensatedRealSum covariance_sum;
  for (std::size_t block = 0; block < block_count; ++block) {
    const std::size_t block_offset = block * mode_count;
    const double left_difference = block_values[block_offset + left_mode] - means[left_mode];
    const double right_difference = block_values[block_offset + right_mode] - means[right_mode];
    const double product = left_difference * right_difference;
    if (!std::isfinite(product)) {
      throw std::overflow_error("density covariance product is non-finite");
    }
    covariance_sum.add(product);
  }
  const auto block_divisor = static_cast<double>(block_count);
  const auto dof_divisor = static_cast<double>(block_count - 1);
  const double covariance = (covariance_sum.value() / block_divisor) / dof_divisor;
  if (!std::isfinite(covariance) || (diagonal && covariance < 0.0)) {
    throw std::overflow_error("density covariance of mean is invalid");
  }
  return covariance;
}

struct DensityCovarianceCoordinates {
  std::size_t row_frequency;
  std::size_t column_frequency;
};

std::size_t density_covariance_index(const std::size_t frequency_count, const std::size_t momentum,
                                     const DensityCovarianceCoordinates coordinates) noexcept {
  return ((((momentum * frequency_count) + coordinates.row_frequency) * frequency_count) +
          coordinates.column_frequency);
}

std::vector<double> density_block_covariances(const MatsubaraModeSet &modes,
                                              const std::span<const double> block_values,
                                              const std::span<const double> means,
                                              const std::size_t block_count) {
  auto covariances = make_checked_vector<double>(density_covariance_extent(modes),
                                                 "density covariances exceed vector capacity");
  const std::size_t frequency_count = modes.frequency_count();
  const std::size_t momentum_count = modes.momentum_count();
  for (std::size_t momentum = 0; momentum < momentum_count; ++momentum) {
    for (std::size_t left_frequency = 0; left_frequency < frequency_count; ++left_frequency) {
      const std::size_t left_mode = (left_frequency * momentum_count) + momentum;
      for (std::size_t right_frequency = left_frequency; right_frequency < frequency_count;
           ++right_frequency) {
        const std::size_t right_mode = (right_frequency * momentum_count) + momentum;
        const double covariance =
            density_covariance_of_mean(block_values, means, modes.mode_count(), block_count,
                                       left_mode, right_mode, left_frequency == right_frequency);
        const std::size_t upper = density_covariance_index(
            frequency_count, momentum,
            {.row_frequency = left_frequency, .column_frequency = right_frequency});
        const std::size_t lower = density_covariance_index(
            frequency_count, momentum,
            {.row_frequency = right_frequency, .column_frequency = left_frequency});
        covariances[upper] = covariance;
        covariances[lower] = covariance;
      }
    }
  }
  return covariances;
}

MatsubaraModeSet validated_hopping_accumulator_modes(const Model &model, MatsubaraModeSet modes) {
  validate_model_mode_geometry(model, modes,
                               "hopping accumulator model and Matsubara geometry differ");
  return modes;
}

std::size_t response_tensor_index(const std::size_t mode, const std::size_t row,
                                  const std::size_t column, const std::size_t dimension) noexcept {
  return ((((mode * dimension) + row) * dimension) + column);
}

void validate_mean_flux_values(const std::span<const std::complex<double>> values) {
  for (const std::complex<double> value : values) {
    if (!is_finite(value)) {
      throw std::invalid_argument("hopping-response mean flux is non-finite");
    }
  }
}

void validate_flux_response_values(const std::size_t mode_count, const std::size_t dimension,
                                   const std::span<const std::complex<double>> values) {
  for (std::size_t mode = 0; mode < mode_count; ++mode) {
    for (std::size_t left = 0; left < dimension; ++left) {
      for (std::size_t right = left; right < dimension; ++right) {
        const std::size_t upper = response_tensor_index(mode, left, right, dimension);
        const std::size_t lower = response_tensor_index(mode, right, left, dimension);
        const std::complex<double> value = values[upper];
        if (!is_finite(value)) {
          throw std::invalid_argument("hopping-response tensor entry is non-finite");
        }
        if (values[lower] != std::conj(value)) {
          throw std::invalid_argument("hopping-response tensor is not Hermitian");
        }
        if (left == right && (value.imag() != 0.0 || value.real() < 0.0)) {
          throw std::invalid_argument("hopping-response diagonal is invalid");
        }
      }
    }
  }
}

void validate_diamagnetic_values(const std::span<const double> values) {
  for (const double value : values) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::invalid_argument("hopping-response diamagnetic value is invalid");
    }
  }
}

void update_hopping_flux_mode(const ContinuousParticleModes &values, const std::size_t frequency,
                              const std::size_t momentum, const std::size_t mode,
                              const std::size_t dimension,
                              const std::span<std::complex<double>> amplitudes,
                              const std::span<std::complex<double>> updated_flux_sums) {
  for (std::size_t axis = 0; axis < dimension; ++axis) {
    const std::complex<double> amplitude = values.flux(frequency, momentum, axis);
    if (!is_finite(amplitude)) {
      throw std::overflow_error("hopping observation flux amplitude is non-finite");
    }
    amplitudes[axis] = amplitude;
    const std::size_t index = (mode * dimension) + axis;
    updated_flux_sums[index] += amplitude;
    if (!is_finite(updated_flux_sums[index])) {
      throw std::overflow_error("hopping mean-flux sum is non-finite");
    }
  }
}

void update_hopping_response_mode(const std::size_t mode, const std::size_t dimension,
                                  const std::span<const std::complex<double>> amplitudes,
                                  const std::span<std::complex<double>> updated_response_sums) {
  for (std::size_t left = 0; left < dimension; ++left) {
    for (std::size_t right = left; right < dimension; ++right) {
      const std::complex<double> product =
          left == right ? std::complex<double>(std::norm(amplitudes[left]), 0.0)
                        : amplitudes[left] * std::conj(amplitudes[right]);
      if (!is_finite(product)) {
        throw std::overflow_error("hopping flux-response observation is non-finite");
      }
      const std::size_t upper = response_tensor_index(mode, left, right, dimension);
      const std::size_t lower = response_tensor_index(mode, right, left, dimension);
      const std::complex<double> updated = updated_response_sums[upper] + product;
      if (!is_finite(updated)) {
        throw std::overflow_error("hopping flux-response sum is non-finite");
      }
      updated_response_sums[upper] = updated;
      updated_response_sums[lower] = std::conj(updated);
    }
  }
}

void validate_projection_geometry(const ContinuousMeasurementContext &context,
                                  const MatsubaraModeSet &modes) {
  if (context.model().beta() != modes.beta() || context.layout() != modes.layout()) {
    throw std::invalid_argument(
        "continuous measurement context and Matsubara plan geometry differ");
  }
}

void validate_projection_geometry(const ContinuousMeasurementContext &context,
                                  const ImaginaryTimeLagSet &lags) {
  if (context.model().beta() != lags.beta() || context.layout() != lags.layout()) {
    throw std::invalid_argument(
        "continuous measurement context and imaginary-time lag plan geometry differ");
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

struct DensityResidenceSeries {
  std::size_t momentum_count;
  std::vector<double> boundaries;
  // Interval-major, followed by momentum request order.
  std::vector<std::complex<double>> values;

  [[nodiscard]] std::size_t interval_count() const noexcept { return boundaries.size() - 1; }

  [[nodiscard]] std::complex<double> value(const std::size_t interval,
                                           const std::size_t momentum) const noexcept {
    return values[(interval * momentum_count) + momentum];
  }
};

struct ShiftedResidenceInterval {
  double begin;
  double end;
  std::size_t source_interval;
};

class ContinuousDensityLagProjector {
public:
  ContinuousDensityLagProjector(const ContinuousMeasurementContext &context,
                                const ImaginaryTimeLagSet &lags,
                                const std::span<const std::complex<double>> site_step_phases)
      : context_(context), lags_(lags), site_step_phases_(site_step_phases),
        particle_count_(context.model().particle_count()), momentum_count_(lags.momentum_count()),
        dimension_(context.model().dimension()),
        particle_phase_extent_(
            detail::checked_product(particle_count_, momentum_count_,
                                    "continuous density-lag phase extent exceeds size_t")),
        current_density_(make_checked_vector<CompensatedComplexSum>(
            momentum_count_, "continuous density-lag state extent exceeds vector capacity")),
        particle_phases_(make_checked_vector<std::complex<double>>(
            particle_phase_extent_,
            "continuous density-lag particle phases exceed vector capacity")),
        zero_momenta_(make_checked_vector<std::uint8_t>(
            momentum_count_, "continuous density-lag zero-momentum extent exceeds capacity")) {
    const std::size_t component_extent = detail::checked_product(
        momentum_count_, dimension_, "continuous density-lag component extent exceeds size_t");
    if (site_step_phases_.size() != component_extent) {
      throw std::logic_error("continuous density-lag plan has inconsistent phase storage");
    }
    initialize_state();
  }

  [[nodiscard]] std::vector<double> project() {
    const DensityResidenceSeries residence = build_residence_series();
    auto overlaps = make_checked_vector<double>(
        lags_.value_count(), "continuous density-lag result extent exceeds vector capacity");
    for (std::size_t lag = 0; lag < lags_.lag_count(); ++lag) {
      const double tau = lags_.lag(lag);
      const std::vector<double> forward = overlaps_at_lag(residence, tau);
      double reflected_tau = tau == 0.0 ? 0.0 : lags_.beta() - tau;
      if (reflected_tau == lags_.beta()) {
        reflected_tau = 0.0;
      }
      const std::vector<double> reflected =
          reflected_tau == tau ? forward : overlaps_at_lag(residence, reflected_tau);
      for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
        double value = (0.5 * forward[momentum]) + (0.5 * reflected[momentum]);
        if (zero_momenta_[momentum] != 0U) {
          value = exact_zero_momentum_overlap();
        }
        if (!std::isfinite(value)) {
          throw std::overflow_error("continuous density-lag overlap is non-finite");
        }
        overlaps[(lag * momentum_count_) + momentum] = value;
      }
    }
    return overlaps;
  }

private:
  void initialize_state() {
    for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
      zero_momenta_[momentum] =
          std::ranges::all_of(lags_.momentum_indices(momentum),
                              [](const std::size_t component) { return component == 0; })
              ? 1U
              : 0U;
    }
    for (std::size_t particle = 0; particle < particle_count_; ++particle) {
      for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
        const std::complex<double> phase = spatial_site_phase(
            lags_.layout(), lags_.momentum_indices(momentum), context_.seam_positions()[particle]);
        particle_phases_[particle_phase_index(particle, momentum)] = phase;
        current_density_[momentum].add(phase);
      }
    }
  }

  [[nodiscard]] DensityResidenceSeries build_residence_series() {
    const std::size_t maximum_interval_count = detail::checked_add_size(
        context_.event_group_count(), 1, "continuous density-lag interval count exceeds size_t");
    const std::size_t maximum_value_count =
        detail::checked_product(maximum_interval_count, momentum_count_,
                                "continuous density-lag residence extent exceeds size_t");
    const std::size_t maximum_boundary_count = detail::checked_add_size(
        maximum_interval_count, 1, "continuous density-lag boundary count exceeds size_t");

    DensityResidenceSeries residence{.momentum_count = momentum_count_};
    if (maximum_boundary_count > residence.boundaries.max_size() ||
        maximum_value_count > residence.values.max_size()) {
      throw std::length_error("continuous density-lag residence data exceed vector capacity");
    }
    residence.boundaries.reserve(maximum_boundary_count);
    residence.values.reserve(maximum_value_count);
    residence.boundaries.push_back(0.0);

    double previous_time = 0.0;
    for (std::size_t group = 0; group < context_.event_group_count(); ++group) {
      const double time = context_.event_time(group);
      append_residence_interval(residence, previous_time, time);
      apply_event_group(group);
      previous_time = time;
    }
    append_residence_interval(residence, previous_time, lags_.beta());
    if (residence.boundaries.size() < 2 || residence.boundaries.back() != lags_.beta()) {
      throw std::logic_error("continuous density-lag residence partition is incomplete");
    }
    return residence;
  }

  void append_residence_interval(DensityResidenceSeries &residence, const double begin,
                                 const double end) const {
    if (end == begin) {
      return;
    }
    if (end < begin || residence.boundaries.empty() || residence.boundaries.back() != begin) {
      throw std::logic_error("continuous density-lag residence partition is inconsistent");
    }
    for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
      const std::complex<double> value = current_density_[momentum].value();
      if (!is_finite(value)) {
        throw std::overflow_error("continuous density-lag residence value is non-finite");
      }
      residence.values.push_back(value);
    }
    residence.boundaries.push_back(end);
  }

  void apply_event_group(const std::size_t group) {
    for (const ContinuousHop &hop : context_.hops_at(group)) {
      const auto particle = static_cast<std::size_t>(hop.particle);
      if (particle >= particle_count_) {
        throw std::logic_error("continuous density-lag hop has an invalid particle label");
      }
      const auto axis = static_cast<std::size_t>(hop.axis);
      for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
        const std::size_t particle_phase = particle_phase_index(particle, momentum);
        const std::complex<double> departure_phase = particle_phases_[particle_phase];
        const std::complex<double> step_phase = site_step_phases_[(momentum * dimension_) + axis];
        const std::complex<double> arrival_phase =
            departure_phase * (hop.direction > 0 ? step_phase : std::conj(step_phase));
        current_density_[momentum].add(arrival_phase - departure_phase);
        particle_phases_[particle_phase] = arrival_phase;
      }
    }
  }

  [[nodiscard]] std::vector<ShiftedResidenceInterval>
  shifted_intervals(const DensityResidenceSeries &residence, const double tau) const {
    if (!std::isfinite(tau) || tau < 0.0 || tau >= lags_.beta()) {
      throw std::logic_error("continuous density-lag shift is outside [0, beta)");
    }
    const std::size_t interval_count = residence.interval_count();
    const auto upper = std::ranges::upper_bound(residence.boundaries, tau);
    const auto source =
        static_cast<std::size_t>(std::distance(residence.boundaries.begin(), upper) - 1);
    if (source >= interval_count) {
      throw std::logic_error("continuous density-lag shift has no residence interval");
    }

    const std::size_t maximum_shifted_count = detail::checked_add_size(
        interval_count, 1, "continuous shifted-interval count exceeds size_t");
    std::vector<ShiftedResidenceInterval> shifted;
    if (maximum_shifted_count > shifted.max_size()) {
      throw std::length_error("continuous shifted intervals exceed vector capacity");
    }
    shifted.reserve(maximum_shifted_count);
    const auto append = [&shifted](const double begin, const double end,
                                   const std::size_t source_interval) {
      if (end < begin || !std::isfinite(begin) || !std::isfinite(end)) {
        throw std::logic_error("continuous shifted residence interval is invalid");
      }
      if (end == begin) {
        return;
      }
      const double expected_begin = shifted.empty() ? 0.0 : shifted.back().end;
      if (begin != expected_begin) {
        throw std::logic_error("continuous shifted residence intervals do not meet");
      }
      shifted.push_back({.begin = begin, .end = end, .source_interval = source_interval});
    };

    append(0.0, residence.boundaries[source + 1] - tau, source);
    for (std::size_t interval = source + 1; interval < interval_count; ++interval) {
      append(residence.boundaries[interval] - tau, residence.boundaries[interval + 1] - tau,
             interval);
    }
    for (std::size_t interval = 0; interval < source; ++interval) {
      append((residence.boundaries[interval] - tau) + lags_.beta(),
             (residence.boundaries[interval + 1] - tau) + lags_.beta(), interval);
    }
    append((residence.boundaries[source] - tau) + lags_.beta(), lags_.beta(), source);
    if (shifted.empty() || shifted.front().begin != 0.0 || shifted.back().end != lags_.beta()) {
      throw std::logic_error("continuous shifted residence partition is incomplete");
    }
    return shifted;
  }

  [[nodiscard]] std::vector<double> overlaps_at_lag(const DensityResidenceSeries &residence,
                                                    const double tau) const {
    const std::vector<ShiftedResidenceInterval> shifted = shifted_intervals(residence, tau);
    auto sums = make_checked_vector<CompensatedRealSum>(
        momentum_count_, "continuous density-lag overlap sums exceed vector capacity");
    std::size_t base = 0;
    std::size_t displaced = 0;
    while (base < residence.interval_count() && displaced < shifted.size()) {
      const double begin = std::max(residence.boundaries[base], shifted[displaced].begin);
      const double end = std::min(residence.boundaries[base + 1], shifted[displaced].end);
      if (end > begin) {
        const double duration = end - begin;
        for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
          const std::complex<double> product =
              residence.value(shifted[displaced].source_interval, momentum) *
              std::conj(residence.value(base, momentum));
          const double contribution = duration * product.real();
          if (!std::isfinite(contribution)) {
            throw std::overflow_error("continuous density-lag contribution is non-finite");
          }
          sums[momentum].add(contribution);
        }
      }
      const double base_end = residence.boundaries[base + 1];
      const double displaced_end = shifted[displaced].end;
      if (base_end <= displaced_end) {
        ++base;
      }
      if (displaced_end <= base_end) {
        ++displaced;
      }
    }
    if (base != residence.interval_count() || displaced != shifted.size()) {
      throw std::logic_error("continuous density-lag intersection did not cover the period");
    }

    auto values = make_checked_vector<double>(
        momentum_count_, "continuous density-lag overlap values exceed vector capacity");
    for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
      values[momentum] = sums[momentum].value();
      if (!std::isfinite(values[momentum])) {
        throw std::overflow_error("continuous density-lag overlap sum is non-finite");
      }
    }
    return values;
  }

  [[nodiscard]] double exact_zero_momentum_overlap() const {
    const auto particle_count = static_cast<double>(particle_count_);
    const double value = (lags_.beta() * particle_count) * particle_count;
    if (!std::isfinite(value)) {
      throw std::overflow_error("continuous zero-momentum density-lag overlap is non-finite");
    }
    return value;
  }

  [[nodiscard]] std::size_t particle_phase_index(const std::size_t particle,
                                                 const std::size_t momentum) const noexcept {
    return (particle * momentum_count_) + momentum;
  }

  const ContinuousMeasurementContext &context_;
  const ImaginaryTimeLagSet &lags_;
  std::span<const std::complex<double>> site_step_phases_;
  std::size_t particle_count_;
  std::size_t momentum_count_;
  std::size_t dimension_;
  std::size_t particle_phase_extent_;
  std::vector<CompensatedComplexSum> current_density_;
  std::vector<std::complex<double>> particle_phases_;
  std::vector<std::uint8_t> zero_momenta_;
};

using PairDensityOccupancyMap = std::unordered_map<SiteId, std::uint64_t, SiteIdHash>;

void add_pair_count(std::uint64_t &pair_count, const std::uint64_t additional) {
  if (pair_count > std::numeric_limits<std::uint64_t>::max() - additional) {
    throw std::overflow_error("continuous on-site pair count exceeds uint64 range");
  }
  pair_count += additional;
}

class ContinuousPairDensityProjector {
public:
  ContinuousPairDensityProjector(const ContinuousMeasurementContext &context,
                                 const MatsubaraModeSet &modes)
      : context_(context), modes_(modes), momentum_count_(modes.momentum_count()),
        frequency_count_(modes.frequency_count()), mode_count_(modes.mode_count()),
        pair_amplitudes_(make_checked_vector<CompensatedComplexSum>(
            mode_count_, "continuous pair-density extent exceeds vector capacity")),
        current_pair_density_(make_checked_vector<CompensatedComplexSum>(
            momentum_count_, "continuous pair-density state exceeds vector capacity")),
        positions_(context.seam_positions().begin(), context.seam_positions().end()),
        zero_momenta_(make_checked_vector<std::uint8_t>(
            momentum_count_, "continuous pair-density zero-momentum extent exceeds capacity")) {
    occupancies_.reserve(context.model().particle_count());
    initialize_zero_momenta();
    initialize_occupancies();
  }

  [[nodiscard]] std::vector<std::complex<double>> project() {
    double previous_time = 0.0;
    for (std::size_t group = 0; group < context_.event_group_count(); ++group) {
      const double time = context_.event_time(group);
      add_residence_interval(previous_time, time);
      apply_event_group(group);
      previous_time = time;
    }
    add_residence_interval(previous_time, modes_.beta());
    return finish();
  }

private:
  void initialize_zero_momenta() {
    for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
      zero_momenta_[momentum] = is_zero_momentum(modes_, momentum) ? 1U : 0U;
    }
  }

  void initialize_occupancies() {
    for (const SiteId position : positions_) {
      auto [entry, inserted] = occupancies_.try_emplace(position, 0);
      static_cast<void>(inserted);
      const std::uint64_t previous_occupancy = entry->second;
      add_pair_count(pair_count_, previous_occupancy);
      add_pair_density_change(position, static_cast<double>(previous_occupancy));
      if (entry->second == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("continuous site occupancy exceeds uint64 range");
      }
      ++entry->second;
    }
  }

  void add_residence_interval(const double begin, const double end) {
    const double contribution = (end - begin) * static_cast<double>(pair_count_);
    pair_overlap_ += contribution;
    if (!std::isfinite(pair_overlap_)) {
      throw std::overflow_error("continuous pair-density zero mode is non-finite");
    }

    for (std::size_t frequency = 0; frequency < frequency_count_; ++frequency) {
      const std::complex<double> interval = detail::matsubara_interval_transform(
          modes_.frequency_index(frequency), begin, end, modes_.beta());
      for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
        pair_amplitudes_[mode_index(frequency, momentum)].add(
            current_pair_density_[momentum].value() * interval);
      }
    }
  }

  void apply_event_group(const std::size_t group) {
    // Occupancy-dependent residence values are sampled only before and after
    // the complete zero-duration group, never from transient hop order.
    for (const ContinuousHop &hop : context_.hops_at(group)) {
      apply_hop(hop);
    }
  }

  void apply_hop(const ContinuousHop &hop) {
    const auto particle = static_cast<std::size_t>(hop.particle);
    if (particle >= positions_.size()) {
      throw std::logic_error("pair-density hop contains an invalid particle label");
    }
    SiteId &position = positions_[particle];
    if (position != hop.departure) {
      throw std::logic_error("pair-density replay disagrees with the shared event sweep");
    }

    auto departure = occupancies_.find(position);
    if (departure == occupancies_.end() || departure->second == 0) {
      throw std::logic_error("pair-density occupancy state is inconsistent");
    }
    const std::uint64_t removed_pairs = departure->second - 1;
    pair_count_ -= removed_pairs;
    add_pair_density_change(position, -static_cast<double>(removed_pairs));
    --departure->second;
    if (departure->second == 0) {
      occupancies_.erase(departure);
    }

    position = hop.arrival;
    auto [arrival, inserted] = occupancies_.try_emplace(position, 0);
    static_cast<void>(inserted);
    const std::uint64_t added_pairs = arrival->second;
    add_pair_count(pair_count_, added_pairs);
    add_pair_density_change(position, static_cast<double>(added_pairs));
    if (arrival->second == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("continuous site occupancy exceeds uint64 range");
    }
    ++arrival->second;
  }

  void add_pair_density_change(const SiteId site, const double coefficient) {
    if (coefficient == 0.0) {
      return;
    }
    for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
      current_pair_density_[momentum].add(coefficient *
                                          detail::matsubara_site_phase(modes_, momentum, site));
    }
  }

  [[nodiscard]] std::vector<std::complex<double>> finish() const {
    auto values = make_checked_vector<std::complex<double>>(
        mode_count_, "continuous pair-density result extent exceeds vector capacity");
    for (std::size_t frequency = 0; frequency < frequency_count_; ++frequency) {
      for (std::size_t momentum = 0; momentum < momentum_count_; ++momentum) {
        const std::size_t mode = mode_index(frequency, momentum);
        std::complex<double> value = pair_amplitudes_[mode].value();
        if (zero_momenta_[momentum] != 0U && modes_.frequency_index(frequency) == 0) {
          value = {pair_overlap_, 0.0};
        }
        if (!is_finite(value)) {
          throw std::overflow_error("continuous pair-density amplitude is non-finite");
        }
        values[mode] = value;
      }
    }
    return values;
  }

  [[nodiscard]] std::size_t mode_index(const std::size_t frequency,
                                       const std::size_t momentum) const noexcept {
    return (frequency * momentum_count_) + momentum;
  }

  const ContinuousMeasurementContext &context_;
  const MatsubaraModeSet &modes_;
  std::size_t momentum_count_;
  std::size_t frequency_count_;
  std::size_t mode_count_;
  std::vector<CompensatedComplexSum> pair_amplitudes_;
  std::vector<CompensatedComplexSum> current_pair_density_;
  std::vector<SiteId> positions_;
  PairDensityOccupancyMap occupancies_;
  std::vector<std::uint8_t> zero_momenta_;
  std::uint64_t pair_count_ = 0;
  double pair_overlap_ = 0.0;
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

ContinuousDensityLagPlan::ContinuousDensityLagPlan(ImaginaryTimeLagSet lags)
    : lags_(std::move(lags)) {
  const std::size_t component_count =
      detail::checked_product(lags_.momentum_count(), lags_.layout().dimension(),
                              "continuous density-lag phase extent exceeds size_t");
  if (component_count > site_step_phases_.max_size()) {
    throw std::length_error("continuous density-lag phase data exceed vector capacity");
  }
  site_step_phases_.reserve(component_count);
  for (std::size_t momentum = 0; momentum < lags_.momentum_count(); ++momentum) {
    for (std::size_t axis = 0; axis < lags_.layout().dimension(); ++axis) {
      site_step_phases_.push_back(std::polar(1.0, -lags_.wavevector_component(momentum, axis)));
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

ContinuousPairDensityModes::ContinuousPairDensityModes(
    Model model, MatsubaraModeSet modes, std::vector<std::complex<double>> pair_density_values)
    : model_(model), pair_density_(std::move(modes), std::move(pair_density_values)) {
  validate_model_mode_geometry(model_, pair_density_.modes(),
                               "continuous pair-density model and Matsubara geometry differ");
  for (const std::complex<double> value : pair_density_.values()) {
    if (!is_finite(value)) {
      throw std::invalid_argument("continuous pair-density amplitude is non-finite");
    }
  }
}

ContinuousDensityLagValues::ContinuousDensityLagValues(Model model, ImaginaryTimeLagSet lags,
                                                       std::vector<double> overlap_values)
    : model_(model), lags_(std::move(lags)), overlap_values_(std::move(overlap_values)) {
  if (model_.beta() != lags_.beta() ||
      TorusLayout(model_.linear_size(), model_.dimension()) != lags_.layout()) {
    throw std::invalid_argument("continuous density-lag geometry does not match its model");
  }
  if (overlap_values_.size() != lags_.value_count()) {
    throw std::invalid_argument("continuous density-lag result has the wrong value extent");
  }
  for (const double value : overlap_values_) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("continuous density-lag overlap is non-finite");
    }
  }
}

double ContinuousDensityLagValues::overlap(const std::size_t lag,
                                           const std::size_t momentum) const {
  static_cast<void>(lags_.lag(lag));
  static_cast<void>(lags_.momentum_indices(momentum));
  return overlap_values_[(lag * lags_.momentum_count()) + momentum];
}

ContinuousMatsubaraDensityCorrelations::ContinuousMatsubaraDensityCorrelations(
    Model model, MatsubaraModeField<double> correlations,
    std::vector<std::complex<double>> mean_amplitudes, const std::size_t sample_count)
    : model_(model), correlations_(std::move(correlations)),
      mean_amplitudes_(std::move(mean_amplitudes)), sample_count_(sample_count) {
  validate_model_mode_geometry(
      model_, modes(), "continuous density-correlation model and Matsubara geometry differ");
  if (mean_amplitudes_.size() != modes().mode_count()) {
    throw std::invalid_argument("continuous density correlations have the wrong mean extent");
  }
  if (sample_count_ == 0) {
    throw std::invalid_argument("continuous density correlations require a nonzero sample count");
  }
  for (const std::complex<double> mean : mean_amplitudes_) {
    if (!is_finite(mean)) {
      throw std::invalid_argument("continuous density mean amplitude is non-finite");
    }
  }
  for (const double correlation : correlations_.values()) {
    if (!std::isfinite(correlation) || correlation < 0.0) {
      throw std::invalid_argument("continuous density susceptibility is invalid");
    }
  }
}

std::complex<double>
ContinuousMatsubaraDensityCorrelations::mean_amplitude(const std::size_t frequency,
                                                       const std::size_t momentum) const {
  static_cast<void>(correlations_.at(frequency, momentum));
  return mean_amplitudes_[(frequency * modes().momentum_count()) + momentum];
}

DensityMatsubaraAccumulator::DensityMatsubaraAccumulator(Model model, MatsubaraModeSet modes)
    : model_(model), modes_(validated_density_accumulator_modes(model_, std::move(modes))),
      amplitude_sums_(make_checked_vector<std::complex<double>>(
          modes_.mode_count(), "density amplitude sums exceed vector capacity")),
      centered_observation_sums_(make_checked_vector<double>(
          modes_.mode_count(), "density susceptibility sums exceed vector capacity")),
      analytic_means_(density_analytic_means(model_, modes_)) {}

void DensityMatsubaraAccumulator::observe(const ContinuousParticleModes &values) {
  const std::size_t updated_sample_count =
      detail::checked_add_size(sample_count_, 1, "density accumulator sample count exceeds size_t");
  const DensityModeObservation observation =
      density_mode_observation(values, model_, modes_, analytic_means_);
  std::vector<std::complex<double>> updated_amplitude_sums = amplitude_sums_;
  std::vector<double> updated_centered_observation_sums = centered_observation_sums_;
  for (std::size_t mode = 0; mode < modes_.mode_count(); ++mode) {
    updated_amplitude_sums[mode] += observation.amplitudes[mode];
    if (!is_finite(updated_amplitude_sums[mode])) {
      throw std::overflow_error("density mean-amplitude sum is non-finite");
    }
    updated_centered_observation_sums[mode] += observation.connected_values[mode];
    if (!std::isfinite(updated_centered_observation_sums[mode])) {
      throw std::overflow_error("density susceptibility sum is non-finite");
    }
  }

  amplitude_sums_.swap(updated_amplitude_sums);
  centered_observation_sums_.swap(updated_centered_observation_sums);
  sample_count_ = updated_sample_count;
}

ContinuousMatsubaraDensityCorrelations DensityMatsubaraAccumulator::finish() const {
  if (sample_count_ == 0) {
    throw std::logic_error("cannot finish a density Matsubara accumulator without samples");
  }

  auto mean_amplitudes = make_checked_vector<std::complex<double>>(
      modes_.mode_count(), "density mean result exceeds vector capacity");
  auto correlations = make_checked_vector<double>(
      modes_.mode_count(), "density susceptibility result exceeds vector capacity");
  const auto sample_divisor = static_cast<double>(sample_count_);
  for (std::size_t mode = 0; mode < modes_.mode_count(); ++mode) {
    mean_amplitudes[mode] = amplitude_sums_[mode] / sample_divisor;
    correlations[mode] = centered_observation_sums_[mode] / sample_divisor;
    if (!is_finite(mean_amplitudes[mode])) {
      throw std::overflow_error("density mean result is non-finite");
    }
    if (!std::isfinite(correlations[mode]) || correlations[mode] < 0.0) {
      throw std::overflow_error("density susceptibility result is non-finite");
    }
  }

  return {model_, MatsubaraModeField<double>(modes_, std::move(correlations)),
          std::move(mean_amplitudes), sample_count_};
}

DensityMatsubaraBlockSeries::DensityMatsubaraBlockSeries(Model model, MatsubaraModeSet modes,
                                                         const std::size_t measurements_per_block,
                                                         std::vector<double> block_values,
                                                         const std::size_t sample_count)
    : model_(model), modes_(validated_density_accumulator_modes(model_, std::move(modes))),
      measurements_per_block_(validated_measurements_per_block(measurements_per_block)),
      sample_count_(sample_count), block_values_(std::move(block_values)) {
  block_count_ =
      validate_density_block_series(modes_, measurements_per_block_, block_values_, sample_count_);
  means_ = density_block_means(modes_, block_values_, block_count_);
  covariances_ = density_block_covariances(modes_, block_values_, means_, block_count_);
}

std::size_t DensityMatsubaraBlockSeries::mode_index(const std::size_t frequency,
                                                    const std::size_t momentum) const {
  if (frequency >= modes_.frequency_count()) {
    throw std::out_of_range("density block-series frequency index is out of range");
  }
  if (momentum >= modes_.momentum_count()) {
    throw std::out_of_range("density block-series momentum index is out of range");
  }
  return (frequency * modes_.momentum_count()) + momentum;
}

std::size_t DensityMatsubaraBlockSeries::covariance_index(const std::size_t momentum,
                                                          const std::size_t left_frequency,
                                                          const std::size_t right_frequency) const {
  if (momentum >= modes_.momentum_count()) {
    throw std::out_of_range("density covariance momentum index is out of range");
  }
  if (left_frequency >= modes_.frequency_count()) {
    throw std::out_of_range("density covariance left-frequency index is out of range");
  }
  if (right_frequency >= modes_.frequency_count()) {
    throw std::out_of_range("density covariance right-frequency index is out of range");
  }
  return density_covariance_index(
      modes_.frequency_count(), momentum,
      {.row_frequency = left_frequency, .column_frequency = right_frequency});
}

double DensityMatsubaraBlockSeries::block_value(const std::size_t block,
                                                const std::size_t frequency,
                                                const std::size_t momentum) const {
  if (block >= block_count_) {
    throw std::out_of_range("density block index is out of range");
  }
  return block_values_[(block * modes_.mode_count()) + mode_index(frequency, momentum)];
}

double DensityMatsubaraBlockSeries::mean(const std::size_t frequency,
                                         const std::size_t momentum) const {
  return means_[mode_index(frequency, momentum)];
}

double DensityMatsubaraBlockSeries::covariance_of_mean(const std::size_t momentum,
                                                       const std::size_t left_frequency,
                                                       const std::size_t right_frequency) const {
  return covariances_[covariance_index(momentum, left_frequency, right_frequency)];
}

double DensityMatsubaraBlockSeries::standard_error(const std::size_t frequency,
                                                   const std::size_t momentum) const {
  const std::size_t covariance = covariance_index(momentum, frequency, frequency);
  return std::sqrt(covariances_[covariance]);
}

double DensityMatsubaraBlockSeries::jackknife_mean(const std::size_t omitted_block,
                                                   const std::size_t frequency,
                                                   const std::size_t momentum) const {
  if (omitted_block >= block_count_) {
    throw std::out_of_range("density jackknife block index is out of range");
  }
  const std::size_t mode = mode_index(frequency, momentum);
  const auto jackknife_divisor = static_cast<double>(block_count_ - 1);
  CompensatedRealSum jackknife_sum;
  for (std::size_t block = 0; block < block_count_; ++block) {
    if (block != omitted_block) {
      jackknife_sum.add(block_values_[(block * modes_.mode_count()) + mode] / jackknife_divisor);
    }
  }
  const double mean_without_block = jackknife_sum.value();
  if (!std::isfinite(mean_without_block) || mean_without_block < 0.0) {
    throw std::overflow_error("density jackknife mean is invalid");
  }
  return mean_without_block;
}

DensityMatsubaraBlockAccumulator::DensityMatsubaraBlockAccumulator(
    Model model, MatsubaraModeSet modes, const std::size_t measurements_per_block)
    : model_(model), modes_(validated_density_accumulator_modes(model_, std::move(modes))),
      measurements_per_block_(validated_measurements_per_block(measurements_per_block)),
      pending_sums_(make_checked_vector<double>(
          modes_.mode_count(), "density pending block sums exceed vector capacity")),
      analytic_means_(density_analytic_means(model_, modes_)) {}

void DensityMatsubaraBlockAccumulator::observe(const ContinuousParticleModes &values) {
  const std::size_t updated_sample_count = detail::checked_add_size(
      sample_count_, 1, "density block accumulator sample count exceeds size_t");
  const std::size_t updated_pending_sample_count = detail::checked_add_size(
      pending_sample_count_, 1, "density pending sample count exceeds size_t");
  if (updated_pending_sample_count > measurements_per_block_) {
    throw std::logic_error("density pending sample count exceeds its block size");
  }
  const DensityModeObservation observation =
      density_mode_observation(values, model_, modes_, analytic_means_);

  std::vector<double> updated_pending_sums = pending_sums_;
  for (std::size_t mode = 0; mode < modes_.mode_count(); ++mode) {
    updated_pending_sums[mode] += observation.connected_values[mode];
    if (!std::isfinite(updated_pending_sums[mode])) {
      throw std::overflow_error("density pending block sum is non-finite");
    }
  }

  if (updated_pending_sample_count < measurements_per_block_) {
    pending_sums_.swap(updated_pending_sums);
    pending_sample_count_ = updated_pending_sample_count;
    sample_count_ = updated_sample_count;
    return;
  }

  auto completed_block = make_checked_vector<double>(
      modes_.mode_count(), "density completed block exceeds vector capacity");
  const auto block_divisor = static_cast<double>(measurements_per_block_);
  for (std::size_t mode = 0; mode < modes_.mode_count(); ++mode) {
    completed_block[mode] = updated_pending_sums[mode] / block_divisor;
    if (!std::isfinite(completed_block[mode]) || completed_block[mode] < 0.0) {
      throw std::overflow_error("density completed block value is invalid");
    }
  }
  const std::size_t updated_block_value_count = detail::checked_add_size(
      block_values_.size(), modes_.mode_count(), "density block-value extent exceeds size_t");
  if (updated_block_value_count > block_values_.max_size()) {
    throw std::length_error("density block values exceed vector capacity");
  }

  block_values_.insert(block_values_.end(), completed_block.begin(), completed_block.end());
  std::ranges::fill(pending_sums_, 0.0);
  pending_sample_count_ = 0;
  sample_count_ = updated_sample_count;
}

DensityMatsubaraBlockSeries DensityMatsubaraBlockAccumulator::finish() const {
  if (pending_sample_count_ != 0) {
    throw std::logic_error("cannot finish a density block accumulator with a partial block");
  }
  if (completed_block_count() < 2) {
    throw std::logic_error("density block accumulator requires at least two complete blocks");
  }
  return {model_, modes_, measurements_per_block_, block_values_, sample_count_};
}

HoppingResponse::HoppingResponse(Model model, MatsubaraModeSet modes,
                                 std::vector<std::complex<double>> mean_flux_values,
                                 std::vector<std::complex<double>> flux_response_values,
                                 std::vector<double> diamagnetic_values,
                                 const std::size_t sample_count)
    : model_(model), modes_(std::move(modes)), mean_flux_values_(std::move(mean_flux_values)),
      flux_response_values_(std::move(flux_response_values)),
      diamagnetic_values_(std::move(diamagnetic_values)), sample_count_(sample_count) {
  validate_model_mode_geometry(model_, modes_,
                               "hopping-response model and Matsubara geometry differ");
  const std::size_t dimension = model_.dimension();
  const std::size_t flux_extent = detail::checked_product(
      modes_.mode_count(), dimension, "hopping-response mean-flux extent exceeds size_t");
  const std::size_t response_extent = detail::checked_product(
      flux_extent, dimension, "hopping-response tensor extent exceeds size_t");
  if (mean_flux_values_.size() != flux_extent) {
    throw std::invalid_argument("hopping response has the wrong mean-flux extent");
  }
  if (flux_response_values_.size() != response_extent) {
    throw std::invalid_argument("hopping response has the wrong tensor extent");
  }
  if (diamagnetic_values_.size() != dimension) {
    throw std::invalid_argument("hopping response has the wrong diamagnetic extent");
  }
  if (sample_count_ == 0) {
    throw std::invalid_argument("hopping response requires a nonzero sample count");
  }
  validate_mean_flux_values(mean_flux_values_);
  validate_flux_response_values(modes_.mode_count(), dimension, flux_response_values_);
  validate_diamagnetic_values(diamagnetic_values_);
}

std::size_t HoppingResponse::flux_index(const std::size_t frequency, const std::size_t momentum,
                                        const std::size_t axis) const {
  if (frequency >= modes_.frequency_count()) {
    throw std::out_of_range("hopping-response frequency index is out of range");
  }
  if (momentum >= modes_.momentum_count()) {
    throw std::out_of_range("hopping-response momentum index is out of range");
  }
  if (axis >= model_.dimension()) {
    throw std::out_of_range("hopping-response axis is out of range");
  }
  const std::size_t mode = (frequency * modes_.momentum_count()) + momentum;
  return (mode * model_.dimension()) + axis;
}

std::size_t HoppingResponse::response_index(const std::size_t frequency, const std::size_t momentum,
                                            const std::size_t left, const std::size_t right) const {
  const std::size_t left_component = flux_index(frequency, momentum, left);
  if (right >= model_.dimension()) {
    throw std::out_of_range("hopping-response right axis is out of range");
  }
  return (left_component * model_.dimension()) + right;
}

std::complex<double> HoppingResponse::mean_flux(const std::size_t frequency,
                                                const std::size_t momentum,
                                                const std::size_t axis) const {
  return mean_flux_values_[flux_index(frequency, momentum, axis)];
}

std::complex<double> HoppingResponse::flux_response(const std::size_t frequency,
                                                    const std::size_t momentum,
                                                    const std::size_t left,
                                                    const std::size_t right) const {
  return flux_response_values_[response_index(frequency, momentum, left, right)];
}

double HoppingResponse::diamagnetic(const std::size_t axis) const {
  if (axis >= model_.dimension()) {
    throw std::out_of_range("hopping-response diamagnetic axis is out of range");
  }
  return diamagnetic_values_[axis];
}

std::complex<double> HoppingResponse::paramagnetic(const std::size_t frequency,
                                                   const std::size_t momentum,
                                                   const std::size_t left,
                                                   const std::size_t right) const {
  std::complex<double> value = -flux_response(frequency, momentum, left, right);
  if (left == right) {
    value += diamagnetic(left);
  }
  return value;
}

HoppingResponseAccumulator::HoppingResponseAccumulator(Model model, MatsubaraModeSet modes)
    : model_(model), modes_(validated_hopping_accumulator_modes(model_, std::move(modes))),
      flux_sums_(make_checked_vector<std::complex<double>>(
          detail::checked_product(modes_.mode_count(), model_.dimension(),
                                  "hopping mean-flux extent exceeds size_t"),
          "hopping mean-flux sums exceed vector capacity")),
      response_sums_(make_checked_vector<std::complex<double>>(
          detail::checked_product(
              detail::checked_product(modes_.mode_count(), model_.dimension(),
                                      "hopping response component extent exceeds size_t"),
              model_.dimension(), "hopping response tensor extent exceeds size_t"),
          "hopping response sums exceed vector capacity")),
      axis_event_count_sums_(make_checked_vector<std::size_t>(
          model_.dimension(), "hopping axis-count sums exceed vector capacity")) {}

void HoppingResponseAccumulator::observe(const ContinuousParticleModes &values) {
  if (values.model() != model_) {
    throw std::invalid_argument("hopping observation has a different model");
  }
  if (values.modes() != modes_) {
    throw std::invalid_argument("hopping observation has different Matsubara modes");
  }

  const std::size_t updated_sample_count =
      detail::checked_add_size(sample_count_, 1, "hopping accumulator sample count exceeds size_t");
  std::vector<std::complex<double>> updated_flux_sums = flux_sums_;
  std::vector<std::complex<double>> updated_response_sums = response_sums_;
  std::vector<std::size_t> updated_axis_event_count_sums = axis_event_count_sums_;
  auto amplitudes = make_checked_vector<std::complex<double>>(
      model_.dimension(), "hopping observation workspace exceeds vector capacity");

  for (std::size_t frequency = 0; frequency < modes_.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes_.momentum_count(); ++momentum) {
      const std::size_t mode = (frequency * modes_.momentum_count()) + momentum;
      update_hopping_flux_mode(values, frequency, momentum, mode, model_.dimension(), amplitudes,
                               updated_flux_sums);
      update_hopping_response_mode(mode, model_.dimension(), amplitudes, updated_response_sums);
    }
  }

  for (std::size_t axis = 0; axis < model_.dimension(); ++axis) {
    updated_axis_event_count_sums[axis] =
        detail::checked_add_size(updated_axis_event_count_sums[axis], values.axis_event_count(axis),
                                 "hopping axis event-count sum exceeds size_t");
  }

  flux_sums_.swap(updated_flux_sums);
  response_sums_.swap(updated_response_sums);
  axis_event_count_sums_.swap(updated_axis_event_count_sums);
  sample_count_ = updated_sample_count;
}

HoppingResponse HoppingResponseAccumulator::finish() const {
  if (sample_count_ == 0) {
    throw std::logic_error("cannot finish a hopping-response accumulator without samples");
  }

  const std::size_t dimension = model_.dimension();
  const std::size_t flux_extent = detail::checked_product(
      modes_.mode_count(), dimension, "hopping mean-flux result extent exceeds size_t");
  const std::size_t response_extent = detail::checked_product(
      flux_extent, dimension, "hopping response result extent exceeds size_t");
  auto mean_flux_values = make_checked_vector<std::complex<double>>(
      flux_extent, "hopping mean-flux result exceeds vector capacity");
  auto flux_response_values = make_checked_vector<std::complex<double>>(
      response_extent, "hopping response result exceeds vector capacity");
  auto diamagnetic_values =
      make_checked_vector<double>(dimension, "hopping diamagnetic result exceeds vector capacity");
  const auto sample_divisor = static_cast<double>(sample_count_);
  const auto volume_divisor = static_cast<double>(model_.volume());

  for (std::size_t index = 0; index < flux_extent; ++index) {
    mean_flux_values[index] = flux_sums_[index] / sample_divisor;
    if (!is_finite(mean_flux_values[index])) {
      throw std::overflow_error("hopping mean-flux result is non-finite");
    }
  }
  for (std::size_t mode = 0; mode < modes_.mode_count(); ++mode) {
    for (std::size_t left = 0; left < dimension; ++left) {
      for (std::size_t right = left; right < dimension; ++right) {
        const std::size_t upper = response_tensor_index(mode, left, right, dimension);
        const std::size_t lower = response_tensor_index(mode, right, left, dimension);
        const std::complex<double> response =
            ((response_sums_[upper] / sample_divisor) / volume_divisor) / model_.beta();
        if (!is_finite(response) ||
            (left == right && (response.imag() != 0.0 || response.real() < 0.0))) {
          throw std::overflow_error("hopping flux-response result is non-finite");
        }
        flux_response_values[upper] = response;
        flux_response_values[lower] = std::conj(response);
      }
    }
  }
  for (std::size_t axis = 0; axis < dimension; ++axis) {
    diamagnetic_values[axis] =
        ((static_cast<double>(axis_event_count_sums_[axis]) / sample_divisor) / volume_divisor) /
        model_.beta();
    if (!std::isfinite(diamagnetic_values[axis]) || diamagnetic_values[axis] < 0.0) {
      throw std::overflow_error("hopping diamagnetic result is non-finite");
    }
  }

  return {model_,
          modes_,
          std::move(mean_flux_values),
          std::move(flux_response_values),
          std::move(diamagnetic_values),
          sample_count_};
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

ContinuousPairDensityModes
continuous_pair_density_modes(const ContinuousMeasurementContext &context,
                              const ContinuousMatsubaraPlan &plan) {
  const MatsubaraModeSet &modes = plan.modes();
  validate_projection_geometry(context, modes);
  ContinuousPairDensityProjector projector(context, modes);
  return {context.model(), modes, projector.project()};
}

ContinuousPairDensityModes
continuous_pair_density_modes(const ContinuousConfiguration &configuration,
                              const ContinuousMatsubaraPlan &plan) {
  return continuous_pair_density_modes(ContinuousMeasurementContext(configuration), plan);
}

ContinuousDensityLagValues
continuous_density_lag_values(const ContinuousMeasurementContext &context,
                              const ContinuousDensityLagPlan &plan) {
  const ImaginaryTimeLagSet &lags = plan.lags();
  validate_projection_geometry(context, lags);
  ContinuousDensityLagProjector projector(context, lags, plan.site_step_phases_);
  return {context.model(), lags, projector.project()};
}

ContinuousDensityLagValues
continuous_density_lag_values(const ContinuousConfiguration &configuration,
                              const ContinuousDensityLagPlan &plan) {
  return continuous_density_lag_values(ContinuousMeasurementContext(configuration), plan);
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
  return spatial_site_phase(modes.layout(), modes.momentum_indices(momentum), site);
}

} // namespace detail

} // namespace qmc
