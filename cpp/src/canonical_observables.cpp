#include "canonical_recursion.hpp"
#include "qmc/free_numerics.hpp"
#include "qmc/observables.hpp"
#include "qmc/torus_layout.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc {
namespace {

struct LogDerivatives {
  double first = 0.0;
  double second = 0.0;
};

LogDerivatives one_particle_beta_derivatives(const Model &model,
                                             const OneParticleSpectrum &spectrum,
                                             const std::size_t length) {
  const double duration = static_cast<double>(length) * model.beta();
  const double scale = 2.0 * spectrum.hopping() * duration;
  if (!std::isfinite(scale)) {
    throw std::overflow_error("one-particle derivative scale overflowed");
  }

  double normalization = 0.0;
  double cosine_sum = 0.0;
  double cosine_square_sum = 0.0;
  for (const double cosine : spectrum.cosines()) {
    const double weight = std::exp(scale * (cosine - 1.0));
    normalization += weight;
    cosine_sum += weight * cosine;
    cosine_square_sum += weight * cosine * cosine;
  }
  const double mean_cosine = cosine_sum / normalization;
  double cosine_variance = (cosine_square_sum / normalization) - (mean_cosine * mean_cosine);
  cosine_variance = std::max(0.0, cosine_variance);

  const auto dimensions = static_cast<double>(model.dimension());
  const double mean_energy = -2.0 * spectrum.hopping() * dimensions * mean_cosine;
  const double energy_variance =
      4.0 * spectrum.hopping() * spectrum.hopping() * dimensions * cosine_variance;
  const auto cycle_length = static_cast<double>(length);
  return LogDerivatives{
      .first = -cycle_length * mean_energy,
      .second = cycle_length * cycle_length * energy_variance,
  };
}

double clamp_nonnegative_roundoff(const double value, const double scale) {
  const double tolerance = 256.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, scale);
  if (value < 0.0 && value >= -tolerance) {
    return 0.0;
  }
  return value;
}

using SecondDerivativeFinalizer = double (*)(double value, double scale);

double preserve_second_derivative(const double value, [[maybe_unused]] const double scale) {
  return value;
}

std::vector<LogDerivatives>
canonical_log_derivatives(const CanonicalEnsemble &ensemble,
                          const std::span<const LogDerivatives> log_cycle_derivatives,
                          const SecondDerivativeFinalizer finalize_second) {
  const auto log_cycle_weights = ensemble.log_cycle_weights();
  const auto log_partitions = ensemble.log_partitions();
  if (log_cycle_derivatives.size() != log_cycle_weights.size() ||
      log_cycle_weights.size() != log_partitions.size()) {
    throw std::logic_error("canonical derivative inputs have inconsistent sizes");
  }

  const auto count = ensemble.model().particle_count();
  std::vector<LogDerivatives> result(count + 1);
  std::vector<double> probabilities(count);
  for (std::size_t particles = 1; particles <= count; ++particles) {
    const double log_particles = std::log(static_cast<double>(particles));
    double probability_sum = 0.0;
    for (std::size_t length = 1; length <= particles; ++length) {
      const double log_probability = log_cycle_weights[length] +
                                     log_partitions[particles - length] - log_particles -
                                     log_partitions[particles];
      probabilities[length - 1] = std::exp(log_probability);
      probability_sum += probabilities[length - 1];
    }
    if (!std::isfinite(probability_sum) || probability_sum <= 0.0) {
      throw std::runtime_error("failed to normalize canonical derivative recursion");
    }

    double first = 0.0;
    double raw_second = 0.0;
    for (std::size_t length = 1; length <= particles; ++length) {
      const double probability = probabilities[length - 1] / probability_sum;
      const double candidate_first =
          log_cycle_derivatives[length].first + result[particles - length].first;
      const double candidate_second =
          log_cycle_derivatives[length].second + result[particles - length].second;
      first += probability * candidate_first;
      raw_second += probability * (candidate_second + (candidate_first * candidate_first));
    }
    result[particles].first = first;
    result[particles].second = finalize_second(raw_second - (first * first), raw_second);
  }
  return result;
}

std::pair<double, double> occupation_moments(const double energy, const CanonicalEnsemble &ensemble,
                                             const std::span<double> terms) {
  const Model &model = ensemble.model();
  const auto log_Z = ensemble.log_partitions();
  const std::size_t particle_count = model.particle_count();
  assert(terms.size() >= particle_count);
  if (particle_count == 0) {
    return {0.0, 0.0};
  }
  for (std::size_t length = 1; length <= particle_count; ++length) {
    const double duration = static_cast<double>(length) * model.beta();
    terms[length - 1] =
        log_Z[particle_count - length] - log_Z[particle_count] - (duration * energy);
  }
  const double occupation = std::exp(log_sum_exp(terms.first(particle_count)));

  double factorial_moment = 0.0;
  if (particle_count >= 2) {
    for (std::size_t length = 2; length <= particle_count; ++length) {
      const double duration = static_cast<double>(length) * model.beta();
      terms[length - 2] = std::log(2.0 * static_cast<double>(length - 1)) +
                          log_Z[particle_count - length] - log_Z[particle_count] -
                          (duration * energy);
    }
    factorial_moment = std::exp(log_sum_exp(terms.first(particle_count - 1)));
  }
  const double second_moment = factorial_moment + occupation;
  const double variance =
      clamp_nonnegative_roundoff(second_moment - (occupation * occupation), second_moment);
  return {occupation, variance};
}

double momentum_coherence_length(const MomentumDistribution &momentum, const Model &model) {
  if (model.linear_size() <= 1 || momentum.condensate_occupation <= 0.0) {
    return 0.0;
  }
  double minimum_momentum_occupation = 0.0;
  std::size_t flat = 1;
  for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
    minimum_momentum_occupation += momentum.modes[flat].occupation;
    flat *= static_cast<std::size_t>(model.linear_size());
  }
  minimum_momentum_occupation /= static_cast<double>(model.dimension());
  if (minimum_momentum_occupation <= 0.0) {
    return 0.0;
  }
  const double occupation_ratio = momentum.condensate_occupation / minimum_momentum_occupation;
  const double ratio = clamp_nonnegative_roundoff(occupation_ratio - 1.0, occupation_ratio);
  const double lattice_momentum = std::numbers::pi / static_cast<double>(model.linear_size());
  return std::sqrt(std::max(0.0, ratio)) / (2.0 * std::sin(lattice_momentum));
}

} // namespace

CanonicalThermodynamics canonical_thermodynamics(const CanonicalEnsemble &ensemble) {
  const Model &model = ensemble.model();
  const OneParticleSpectrum &spectrum = ensemble.spectrum();
  const auto log_Z = ensemble.log_partitions();
  if (model.beta() <= 0.0) {
    throw std::invalid_argument("canonical thermodynamics requires beta > 0");
  }

  const auto count = model.particle_count();
  std::vector<LogDerivatives> log_z_derivatives(count + 1);
  for (std::size_t length = 1; length <= count; ++length) {
    log_z_derivatives[length] = one_particle_beta_derivatives(model, spectrum, length);
  }
  const auto log_Z_derivatives =
      canonical_log_derivatives(ensemble, log_z_derivatives, clamp_nonnegative_roundoff);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  CanonicalThermodynamics result{
      .free_energy = std::vector<double>(count + 1),
      .energy = std::vector<double>(count + 1),
      .heat_capacity = std::vector<double>(count + 1),
      .entropy = std::vector<double>(count + 1),
      .addition_chemical_potential = std::vector<double>(count + 1, nan),
  };
  for (std::size_t particles = 0; particles <= count; ++particles) {
    result.free_energy[particles] = -log_Z[particles] / model.beta();
    result.energy[particles] = -log_Z_derivatives[particles].first;
    result.heat_capacity[particles] =
        model.beta() * model.beta() * log_Z_derivatives[particles].second;
    result.entropy[particles] = log_Z[particles] + (model.beta() * result.energy[particles]);
    if (particles > 0) {
      result.addition_chemical_potential[particles] =
          result.free_energy[particles] - result.free_energy[particles - 1];
    }
  }
  return result;
}

CanonicalThermodynamics canonical_thermodynamics(const Model &model) {
  return canonical_thermodynamics(CanonicalEnsemble(model));
}

MomentumDistribution momentum_distribution(const CanonicalEnsemble &ensemble) {
  const Model &model = ensemble.model();
  const OneParticleSpectrum &spectrum = ensemble.spectrum();
  const TorusLayout &layout = spectrum.layout();
  const auto wavevectors = spectrum.wavevectors();
  const auto volume = layout.volume();
  MomentumDistribution result;
  result.modes.reserve(volume);
  std::vector<double> occupation_moment_terms(model.particle_count());

  for (std::size_t flat = 0; flat < volume; ++flat) {
    MomentumMode mode;
    mode.indices = layout.decode(SiteId(flat));
    mode.wavevector.resize(model.dimension());
    for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
      mode.wavevector[axis] = wavevectors[mode.indices[axis]];
    }
    mode.energy = spectrum.energy(mode.indices);

    const auto [occupation, variance] =
        occupation_moments(mode.energy, ensemble, occupation_moment_terms);
    mode.occupation = occupation;
    mode.occupation_variance = variance;
    result.kinetic_energy += mode.energy * mode.occupation;
    result.modes.push_back(std::move(mode));
  }

  if (!result.modes.empty()) {
    result.condensate_occupation = result.modes.front().occupation;
  }
  if (model.particle_count() > 0) {
    result.condensate_fraction =
        result.condensate_occupation / static_cast<double>(model.particle_count());
  }
  result.condensate_density = result.condensate_occupation / static_cast<double>(volume);
  result.coherence_length = momentum_coherence_length(result, model);
  return result;
}

MomentumDistribution momentum_distribution(const Model &model) {
  return momentum_distribution(CanonicalEnsemble(model));
}

std::vector<OneBodyDensityPoint> one_body_density_matrix(const CanonicalEnsemble &ensemble) {
  const Model &model = ensemble.model();
  const OneParticleSpectrum &spectrum = ensemble.spectrum();
  const auto log_z = ensemble.log_cycle_weights();
  const auto log_Z = ensemble.log_partitions();
  const TorusLayout &layout = spectrum.layout();
  const auto volume = layout.volume();
  const auto linear_size = static_cast<std::size_t>(model.linear_size());
  const auto cosines = spectrum.cosines();

  std::vector<std::vector<double>> kernel_ratios(model.particle_count() + 1,
                                                 std::vector<double>(linear_size));
  for (std::size_t length = 1; length <= model.particle_count(); ++length) {
    const double duration = static_cast<double>(length) * model.beta();
    const double scale = 2.0 * spectrum.hopping() * duration;
    std::vector<double> weights(linear_size);
    double scaled_trace = 0.0;
    for (std::size_t momentum = 0; momentum < linear_size; ++momentum) {
      weights[momentum] = std::exp(scale * (cosines[momentum] - 1.0));
      scaled_trace += weights[momentum];
    }
    for (std::size_t displacement = 0; displacement < linear_size; ++displacement) {
      double numerator = 0.0;
      std::size_t phase_index = 0;
      for (std::size_t momentum = 0; momentum < linear_size; ++momentum) {
        numerator += weights[momentum] * cosines[phase_index];
        phase_index = phase_index >= linear_size - displacement
                          ? phase_index - (linear_size - displacement)
                          : phase_index + displacement;
      }
      const double ratio = numerator / (static_cast<double>(linear_size) * scaled_trace);
      kernel_ratios[length][displacement] = std::max(0.0, ratio);
    }
  }

  std::vector<double> cycle_weights(model.particle_count() + 1);
  for (std::size_t length = 1; length <= model.particle_count(); ++length) {
    cycle_weights[length] = std::exp(log_z[length] + log_Z[model.particle_count() - length] -
                                     log_Z[model.particle_count()]);
  }

  std::vector<OneBodyDensityPoint> result;
  result.reserve(volume);
  for (std::size_t flat = 0; flat < volume; ++flat) {
    const auto components = layout.decode(SiteId(flat));
    OneBodyDensityPoint point{.displacement = Site(model.dimension()), .value = 0.0};
    for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
      point.displacement[axis] = static_cast<Coord>(components[axis]);
    }
    for (std::size_t length = 1; length <= model.particle_count(); ++length) {
      double normalized_kernel = 1.0;
      for (const std::size_t component : components) {
        normalized_kernel *= kernel_ratios[length][component];
      }
      point.value += cycle_weights[length] * normalized_kernel;
    }
    result.push_back(std::move(point));
  }
  return result;
}

std::vector<OneBodyDensityPoint> one_body_density_matrix(const Model &model) {
  return one_body_density_matrix(CanonicalEnsemble(model));
}

ExactCycleStatistics exact_cycle_statistics(const CanonicalEnsemble &ensemble) {
  return exact_cycle_statistics(ensemble, ensemble.model().particle_count());
}

ExactCycleStatistics exact_cycle_statistics(const CanonicalEnsemble &ensemble,
                                            const std::size_t particle_count) {
  if (particle_count > ensemble.model().particle_count()) {
    throw std::out_of_range("cycle-statistics particle count exceeds the ensemble capacity");
  }
  const auto log_z = ensemble.log_cycle_weights();
  const auto log_Z = ensemble.log_partitions();
  ExactCycleStatistics result{
      .expected_cycle_count = std::vector<double>(particle_count + 1),
      .expected_particles = std::vector<double>(particle_count + 1),
      .particle_probability = std::vector<double>(particle_count + 1),
  };
  for (std::size_t length = 1; length <= particle_count; ++length) {
    result.expected_cycle_count[length] =
        std::exp(log_z[length] + log_Z[particle_count - length] - log_Z[particle_count] -
                 std::log(static_cast<double>(length)));
    result.expected_particles[length] =
        static_cast<double>(length) * result.expected_cycle_count[length];
    if (particle_count > 0) {
      result.particle_probability[length] =
          result.expected_particles[length] / static_cast<double>(particle_count);
    }
  }
  return result;
}

ExactCycleStatistics exact_cycle_statistics(const Model &model) {
  return exact_cycle_statistics(CanonicalEnsemble(model));
}

double log_canonical_partition_twisted(const CanonicalEnsemble &ensemble,
                                       std::span<const double> twist) {
  const Model &model = ensemble.model();
  const OneParticleSpectrum &spectrum = ensemble.spectrum();
  if (twist.size() != model.dimension()) {
    throw std::invalid_argument("twist vector has the wrong dimension");
  }
  for (const double component : twist) {
    if (!std::isfinite(component)) {
      throw std::invalid_argument("twist components must be finite");
    }
  }

  const auto count = model.particle_count();
  const double negative_infinity = -std::numeric_limits<double>::infinity();
  std::vector<double> log_z(count + 1, negative_infinity);
  std::vector<double> exponents(spectrum.cosines().size());
  std::vector<double> twist_cosines(model.dimension());
  std::vector<double> twist_sines(model.dimension());
  const double inverse_size = 1.0 / static_cast<double>(model.linear_size());
  for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
    const double shift = twist[axis] * inverse_size;
    twist_cosines[axis] = std::cos(shift);
    twist_sines[axis] = std::sin(shift);
  }
  const auto cosines = spectrum.cosines();
  const auto sines = spectrum.sines();
  for (std::size_t length = 1; length <= count; ++length) {
    const double duration = static_cast<double>(length) * model.beta();
    const double scale = 2.0 * spectrum.hopping() * duration;
    if (!std::isfinite(scale)) {
      throw std::overflow_error("twisted trace exponent scale overflowed");
    }
    double value = 0.0;
    for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
      for (std::size_t momentum = 0; momentum < cosines.size(); ++momentum) {
        const double shifted_cosine =
            (cosines[momentum] * twist_cosines[axis]) - (sines[momentum] * twist_sines[axis]);
        exponents[momentum] = scale * shifted_cosine;
      }
      value += log_sum_exp(exponents);
    }
    log_z[length] = value;
  }
  return detail::canonical_log_partitions(log_z).back();
}

double log_canonical_partition_twisted(const Model &model, std::span<const double> twist) {
  return log_canonical_partition_twisted(CanonicalEnsemble(model), twist);
}

double twist_free_energy_curvature(const CanonicalEnsemble &ensemble, const std::size_t axis) {
  const Model &model = ensemble.model();
  const OneParticleSpectrum &spectrum = ensemble.spectrum();
  if (model.beta() <= 0.0) {
    throw std::invalid_argument("twist free-energy curvature requires beta > 0");
  }
  if (axis >= model.dimension()) {
    throw std::out_of_range("twist axis is out of range");
  }

  std::vector<LogDerivatives> log_z_derivatives(model.particle_count() + 1);
  const double inverse_size = 1.0 / static_cast<double>(model.linear_size());
  const auto cosines = spectrum.cosines();
  const auto sines = spectrum.sines();
  for (std::size_t length = 1; length <= model.particle_count(); ++length) {
    const double duration = static_cast<double>(length) * model.beta();
    const double scale = 2.0 * spectrum.hopping() * duration;
    double normalization = 0.0;
    double sine_sum = 0.0;
    double sine_square_sum = 0.0;
    double cosine_sum = 0.0;
    for (std::size_t momentum = 0; momentum < cosines.size(); ++momentum) {
      const double weight = std::exp(scale * (cosines[momentum] - 1.0));
      normalization += weight;
      sine_sum += weight * sines[momentum];
      sine_square_sum += weight * sines[momentum] * sines[momentum];
      cosine_sum += weight * cosines[momentum];
    }
    const double mean_sine = sine_sum / normalization;
    const double mean_sine_square = sine_square_sum / normalization;
    const double mean_cosine = cosine_sum / normalization;
    log_z_derivatives[length].first = -scale * inverse_size * mean_sine;
    log_z_derivatives[length].second =
        inverse_size * inverse_size *
        (-scale * mean_cosine + scale * scale * (mean_sine_square - (mean_sine * mean_sine)));
  }

  const auto log_Z_derivatives =
      canonical_log_derivatives(ensemble, log_z_derivatives, preserve_second_derivative);
  const double curvature = -log_Z_derivatives[model.particle_count()].second / model.beta();
  return clamp_nonnegative_roundoff(curvature, std::abs(curvature));
}

double twist_free_energy_curvature(const Model &model, const std::size_t axis) {
  return twist_free_energy_curvature(CanonicalEnsemble(model), axis);
}

} // namespace qmc
