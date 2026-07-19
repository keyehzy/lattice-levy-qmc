#include "qmc/observables.hpp"

#include "qmc/free_numerics.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc {
namespace {

void validate_table(const FreeBosonTable &table, const std::size_t particle_count) {
  if (particle_count == std::numeric_limits<std::size_t>::max() ||
      table.log_z.size() < particle_count + 1 || table.log_Z.size() < particle_count + 1) {
    throw std::invalid_argument("canonical table is too short");
  }
  if (table.log_Z[0] != 0.0) {
    throw std::invalid_argument("canonical table must have log_Z[0] == 0");
  }
  for (std::size_t particles = 0; particles <= particle_count; ++particles) {
    if (!std::isfinite(table.log_Z[particles])) {
      throw std::invalid_argument("canonical table contains a non-finite log_Z value");
    }
  }
  for (std::size_t length = 1; length <= particle_count; ++length) {
    if (!std::isfinite(table.log_z[length])) {
      throw std::invalid_argument("canonical table contains a non-finite log_z value");
    }
  }
}

std::vector<std::size_t> decode_flat_index(std::size_t flat, const Model &model) {
  const auto linear_size = static_cast<std::size_t>(model.linear_size);
  std::vector<std::size_t> components(model.dimension);
  for (std::size_t axis = 0; axis < model.dimension; ++axis) {
    components[axis] = flat % linear_size;
    flat /= linear_size;
  }
  return components;
}

std::size_t encode_components(std::span<const std::size_t> components, const Model &model) {
  if (components.size() != model.dimension) {
    throw std::invalid_argument("component vector has the wrong dimension");
  }
  const auto linear_size = static_cast<std::size_t>(model.linear_size);
  std::size_t flat = 0;
  std::size_t stride = 1;
  for (const std::size_t component : components) {
    if (component >= linear_size) {
      throw std::invalid_argument("torus component is outside [0, L)");
    }
    flat += component * stride;
    stride *= linear_size;
  }
  return flat;
}

std::size_t encode_site(const DenseWorldlines &worldlines, const ParticleId particle,
                        const std::size_t time, const Model &model) {
  const auto linear_size = static_cast<std::size_t>(model.linear_size);
  std::size_t flat = 0;
  std::size_t stride = 1;
  for (std::size_t axis = 0; axis < model.dimension; ++axis) {
    const Coord coordinate = worldlines.at(particle, time, axis);
    if (coordinate < 0 || coordinate >= model.linear_size) {
      throw std::logic_error("world-line coordinate lies outside the torus");
    }
    flat += static_cast<std::size_t>(coordinate) * stride;
    stride *= linear_size;
  }
  return flat;
}

std::size_t displacement_index(const std::size_t origin_flat, const std::size_t target_flat,
                               const Model &model) {
  const auto origin = decode_flat_index(origin_flat, model);
  const auto target = decode_flat_index(target_flat, model);
  const auto linear_size = static_cast<std::size_t>(model.linear_size);
  std::vector<std::size_t> displacement(model.dimension);
  for (std::size_t axis = 0; axis < model.dimension; ++axis) {
    displacement[axis] = (target[axis] + linear_size - origin[axis]) % linear_size;
  }
  return encode_components(displacement, model);
}

std::vector<std::vector<std::size_t>>
retained_positions(const IdealBosonConfiguration &configuration) {
  const auto time_points = configuration.time_links_per_beta;
  std::vector<std::vector<std::size_t>> positions(
      time_points, std::vector<std::size_t>(configuration.model.particle_count));
  for (std::size_t time = 0; time < time_points; ++time) {
    for (std::size_t particle = 0; particle < configuration.model.particle_count; ++particle) {
      positions[time][particle] = encode_site(
          configuration.worldlines, static_cast<ParticleId>(particle), time, configuration.model);
    }
  }
  return positions;
}

struct LogDerivatives {
  double first = 0.0;
  double second = 0.0;
};

LogDerivatives one_particle_beta_derivatives(const Model &model, const std::size_t length) {
  const double duration = static_cast<double>(length) * model.beta;
  const double scale = 2.0 * model.hopping * duration;
  if (!std::isfinite(scale)) {
    throw std::overflow_error("one-particle derivative scale overflowed");
  }

  double normalization = 0.0;
  double cosine_sum = 0.0;
  double cosine_square_sum = 0.0;
  for (Coord momentum = 0; momentum < model.linear_size; ++momentum) {
    const double angle = 2.0 * std::numbers::pi * static_cast<double>(momentum) /
                         static_cast<double>(model.linear_size);
    const double cosine = std::cos(angle);
    const double weight = std::exp(scale * (cosine - 1.0));
    normalization += weight;
    cosine_sum += weight * cosine;
    cosine_square_sum += weight * cosine * cosine;
  }
  const double mean_cosine = cosine_sum / normalization;
  double cosine_variance = (cosine_square_sum / normalization) - (mean_cosine * mean_cosine);
  cosine_variance = std::max(0.0, cosine_variance);

  const auto dimensions = static_cast<double>(model.dimension);
  const double mean_energy = -2.0 * model.hopping * dimensions * mean_cosine;
  const double energy_variance = 4.0 * model.hopping * model.hopping * dimensions * cosine_variance;
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

double mode_energy(const std::vector<std::size_t> &indices, const Model &model) {
  double cosine_sum = 0.0;
  for (const std::size_t index : indices) {
    const double angle = 2.0 * std::numbers::pi * static_cast<double>(index) /
                         static_cast<double>(model.linear_size);
    cosine_sum += std::cos(angle);
  }
  const double energy = -2.0 * model.hopping * cosine_sum;
  if (!std::isfinite(energy)) {
    throw std::overflow_error("one-particle energy overflowed");
  }
  return energy;
}

double phase_for_indices(std::span<const std::size_t> left, std::span<const std::size_t> right,
                         const Model &model) {
  double phase = 0.0;
  for (std::size_t axis = 0; axis < model.dimension; ++axis) {
    phase += 2.0 * std::numbers::pi * static_cast<double>(left[axis]) *
             static_cast<double>(right[axis]) / static_cast<double>(model.linear_size);
  }
  return phase;
}

std::pair<double, double> occupation_moments(const double energy, const Model &model,
                                             const FreeBosonTable &table) {
  if (model.particle_count == 0) {
    return {0.0, 0.0};
  }
  std::vector<double> occupation_terms(model.particle_count);
  for (std::size_t length = 1; length <= model.particle_count; ++length) {
    const double duration = static_cast<double>(length) * model.beta;
    occupation_terms[length - 1] = table.log_Z[model.particle_count - length] -
                                   table.log_Z[model.particle_count] - (duration * energy);
  }
  const double occupation = std::exp(log_sum_exp(occupation_terms));

  double factorial_moment = 0.0;
  if (model.particle_count >= 2) {
    std::vector<double> factorial_terms(model.particle_count - 1);
    for (std::size_t length = 2; length <= model.particle_count; ++length) {
      const double duration = static_cast<double>(length) * model.beta;
      factorial_terms[length - 2] = std::log(2.0 * static_cast<double>(length - 1)) +
                                    table.log_Z[model.particle_count - length] -
                                    table.log_Z[model.particle_count] - (duration * energy);
    }
    factorial_moment = std::exp(log_sum_exp(factorial_terms));
  }
  const double second_moment = factorial_moment + occupation;
  const double variance =
      clamp_nonnegative_roundoff(second_moment - (occupation * occupation), second_moment);
  return {occupation, variance};
}

double momentum_coherence_length(const MomentumDistribution &momentum, const Model &model) {
  if (model.linear_size <= 1 || momentum.condensate_occupation <= 0.0) {
    return 0.0;
  }
  double minimum_momentum_occupation = 0.0;
  std::size_t flat = 1;
  for (std::size_t axis = 0; axis < model.dimension; ++axis) {
    minimum_momentum_occupation += momentum.modes[flat].occupation;
    flat *= static_cast<std::size_t>(model.linear_size);
  }
  minimum_momentum_occupation /= static_cast<double>(model.dimension);
  if (minimum_momentum_occupation <= 0.0) {
    return 0.0;
  }
  const double occupation_ratio = momentum.condensate_occupation / minimum_momentum_occupation;
  const double ratio = clamp_nonnegative_roundoff(occupation_ratio - 1.0, occupation_ratio);
  const double lattice_momentum = std::numbers::pi / static_cast<double>(model.linear_size);
  return std::sqrt(std::max(0.0, ratio)) / (2.0 * std::sin(lattice_momentum));
}

void accumulate_pair_counts(std::span<const std::size_t> positions, const Model &model,
                            std::vector<double> &pair_counts) {
  for (const std::size_t origin : positions) {
    for (const std::size_t target : positions) {
      ++pair_counts[displacement_index(origin, target, model)];
    }
  }
}

void accumulate_structure_factor(std::span<const std::size_t> positions, const Model &model,
                                 std::vector<double> &structure_factor) {
  if (model.particle_count == 0) {
    return;
  }
  for (std::size_t momentum = 0; momentum < structure_factor.size(); ++momentum) {
    const auto momentum_components = decode_flat_index(momentum, model);
    std::complex<double> density_mode{0.0, 0.0};
    for (const std::size_t site : positions) {
      const auto site_components = decode_flat_index(site, model);
      const double phase = phase_for_indices(momentum_components, site_components, model);
      density_mode += std::polar(1.0, -phase);
    }
    structure_factor[momentum] +=
        std::norm(density_mode) / static_cast<double>(model.particle_count);
  }
}

} // namespace

CanonicalThermodynamics canonical_thermodynamics(const Model &model, const FreeBosonTable &table) {
  model.validate();
  if (model.beta <= 0.0) {
    throw std::invalid_argument("canonical thermodynamics requires beta > 0");
  }
  validate_table(table, model.particle_count);

  const auto count = model.particle_count;
  std::vector<LogDerivatives> log_z_derivatives(count + 1);
  for (std::size_t length = 1; length <= count; ++length) {
    log_z_derivatives[length] = one_particle_beta_derivatives(model, length);
  }

  std::vector<double> log_Z_first(count + 1, 0.0);
  std::vector<double> log_Z_second(count + 1, 0.0);
  for (std::size_t particles = 1; particles <= count; ++particles) {
    std::vector<double> probabilities(particles);
    double probability_sum = 0.0;
    for (std::size_t length = 1; length <= particles; ++length) {
      const double log_probability = table.log_z[length] + table.log_Z[particles - length] -
                                     std::log(static_cast<double>(particles)) -
                                     table.log_Z[particles];
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
          log_z_derivatives[length].first + log_Z_first[particles - length];
      const double candidate_second =
          log_z_derivatives[length].second + log_Z_second[particles - length];
      first += probability * candidate_first;
      raw_second += probability * (candidate_second + (candidate_first * candidate_first));
    }
    log_Z_first[particles] = first;
    log_Z_second[particles] = clamp_nonnegative_roundoff(raw_second - (first * first), raw_second);
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  CanonicalThermodynamics result{
      .free_energy = std::vector<double>(count + 1),
      .energy = std::vector<double>(count + 1),
      .heat_capacity = std::vector<double>(count + 1),
      .entropy = std::vector<double>(count + 1),
      .addition_chemical_potential = std::vector<double>(count + 1, nan),
  };
  for (std::size_t particles = 0; particles <= count; ++particles) {
    result.free_energy[particles] = -table.log_Z[particles] / model.beta;
    result.energy[particles] = -log_Z_first[particles];
    result.heat_capacity[particles] = model.beta * model.beta * log_Z_second[particles];
    result.entropy[particles] = table.log_Z[particles] + (model.beta * result.energy[particles]);
    if (particles > 0) {
      result.addition_chemical_potential[particles] =
          result.free_energy[particles] - result.free_energy[particles - 1];
    }
  }
  return result;
}

MomentumDistribution momentum_distribution(const Model &model, const FreeBosonTable &table) {
  model.validate();
  validate_table(table, model.particle_count);
  const auto volume = model.volume();
  MomentumDistribution result;
  result.modes.reserve(volume);

  for (std::size_t flat = 0; flat < volume; ++flat) {
    MomentumMode mode;
    mode.indices = decode_flat_index(flat, model);
    mode.wavevector.resize(model.dimension);
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      mode.wavevector[axis] = 2.0 * std::numbers::pi * static_cast<double>(mode.indices[axis]) /
                              static_cast<double>(model.linear_size);
    }
    mode.energy = mode_energy(mode.indices, model);

    const auto [occupation, variance] = occupation_moments(mode.energy, model, table);
    mode.occupation = occupation;
    mode.occupation_variance = variance;
    result.kinetic_energy += mode.energy * mode.occupation;
    result.modes.push_back(std::move(mode));
  }

  if (!result.modes.empty()) {
    result.condensate_occupation = result.modes.front().occupation;
  }
  if (model.particle_count > 0) {
    result.condensate_fraction =
        result.condensate_occupation / static_cast<double>(model.particle_count);
  }
  result.condensate_density = result.condensate_occupation / static_cast<double>(volume);
  result.coherence_length = momentum_coherence_length(result, model);
  return result;
}

std::vector<OneBodyDensityPoint> one_body_density_matrix(const Model &model,
                                                         const FreeBosonTable &table) {
  model.validate();
  validate_table(table, model.particle_count);
  const auto volume = model.volume();
  const auto linear_size = static_cast<std::size_t>(model.linear_size);

  std::vector<std::vector<double>> kernel_ratios(model.particle_count + 1,
                                                 std::vector<double>(linear_size));
  for (std::size_t length = 1; length <= model.particle_count; ++length) {
    const double duration = static_cast<double>(length) * model.beta;
    const double scale = 2.0 * model.hopping * duration;
    std::vector<double> weights(linear_size);
    double scaled_trace = 0.0;
    for (std::size_t momentum = 0; momentum < linear_size; ++momentum) {
      const double angle = 2.0 * std::numbers::pi * static_cast<double>(momentum) /
                           static_cast<double>(model.linear_size);
      weights[momentum] = std::exp(scale * (std::cos(angle) - 1.0));
      scaled_trace += weights[momentum];
    }
    for (std::size_t displacement = 0; displacement < linear_size; ++displacement) {
      double numerator = 0.0;
      for (std::size_t momentum = 0; momentum < linear_size; ++momentum) {
        const double angle = 2.0 * std::numbers::pi * static_cast<double>(momentum) /
                             static_cast<double>(model.linear_size);
        numerator += weights[momentum] * std::cos(angle * static_cast<double>(displacement));
      }
      const double ratio = numerator / (static_cast<double>(linear_size) * scaled_trace);
      kernel_ratios[length][displacement] = std::max(0.0, ratio);
    }
  }

  std::vector<double> cycle_weights(model.particle_count + 1);
  for (std::size_t length = 1; length <= model.particle_count; ++length) {
    cycle_weights[length] =
        std::exp(table.log_z[length] + table.log_Z[model.particle_count - length] -
                 table.log_Z[model.particle_count]);
  }

  std::vector<OneBodyDensityPoint> result;
  result.reserve(volume);
  for (std::size_t flat = 0; flat < volume; ++flat) {
    const auto components = decode_flat_index(flat, model);
    OneBodyDensityPoint point{.displacement = Site(model.dimension), .value = 0.0};
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      point.displacement[axis] = static_cast<Coord>(components[axis]);
    }
    for (std::size_t length = 1; length <= model.particle_count; ++length) {
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

ExactCycleStatistics exact_cycle_statistics(const std::size_t particle_count,
                                            const FreeBosonTable &table) {
  validate_table(table, particle_count);
  ExactCycleStatistics result{
      .expected_cycle_count = std::vector<double>(particle_count + 1),
      .expected_particles = std::vector<double>(particle_count + 1),
      .particle_probability = std::vector<double>(particle_count + 1),
  };
  for (std::size_t length = 1; length <= particle_count; ++length) {
    result.expected_cycle_count[length] =
        std::exp(table.log_z[length] + table.log_Z[particle_count - length] -
                 table.log_Z[particle_count] - std::log(static_cast<double>(length)));
    result.expected_particles[length] =
        static_cast<double>(length) * result.expected_cycle_count[length];
    if (particle_count > 0) {
      result.particle_probability[length] =
          result.expected_particles[length] / static_cast<double>(particle_count);
    }
  }
  return result;
}

std::vector<std::size_t> sampled_cycle_histogram(const IdealBosonConfiguration &configuration) {
  std::vector<std::size_t> result(configuration.model.particle_count + 1);
  for (const IdealCyclePath &cycle : configuration.cycles) {
    if (cycle.labels.empty() || cycle.labels.size() > configuration.model.particle_count) {
      throw std::logic_error("configuration contains an invalid cycle length");
    }
    ++result[cycle.labels.size()];
  }
  return result;
}

std::size_t longest_cycle_length(const IdealBosonConfiguration &configuration) {
  std::size_t result = 0;
  for (const IdealCyclePath &cycle : configuration.cycles) {
    result = std::max(result, cycle.labels.size());
  }
  return result;
}

Site total_winding(const IdealBosonConfiguration &configuration) {
  Site result(configuration.model.dimension);
  for (const IdealCyclePath &cycle : configuration.cycles) {
    if (cycle.winding.size() != configuration.model.dimension) {
      throw std::logic_error("configuration contains a winding vector of the wrong dimension");
    }
    for (std::size_t axis = 0; axis < configuration.model.dimension; ++axis) {
      const Coord value = cycle.winding[axis];
      if ((value > 0 && result[axis] > std::numeric_limits<Coord>::max() - value) ||
          (value < 0 && result[axis] < std::numeric_limits<Coord>::min() - value)) {
        throw std::overflow_error("total winding exceeds the coordinate range");
      }
      result[axis] += value;
    }
  }
  return result;
}

double log_canonical_partition_twisted(const Model &model, std::span<const double> twist) {
  model.validate();
  if (twist.size() != model.dimension) {
    throw std::invalid_argument("twist vector has the wrong dimension");
  }
  for (const double component : twist) {
    if (!std::isfinite(component)) {
      throw std::invalid_argument("twist components must be finite");
    }
  }

  const auto count = model.particle_count;
  const double negative_infinity = -std::numeric_limits<double>::infinity();
  std::vector<double> log_z(count + 1, negative_infinity);
  std::vector<double> log_Z(count + 1, negative_infinity);
  log_Z[0] = 0.0;
  for (std::size_t length = 1; length <= count; ++length) {
    const double duration = static_cast<double>(length) * model.beta;
    const double scale = 2.0 * model.hopping * duration;
    if (!std::isfinite(scale)) {
      throw std::overflow_error("twisted trace exponent scale overflowed");
    }
    double value = 0.0;
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      std::vector<double> exponents(static_cast<std::size_t>(model.linear_size));
      for (Coord momentum = 0; momentum < model.linear_size; ++momentum) {
        const double angle =
            (2.0 * std::numbers::pi * static_cast<double>(momentum) + twist[axis]) /
            static_cast<double>(model.linear_size);
        exponents[static_cast<std::size_t>(momentum)] = scale * std::cos(angle);
      }
      value += log_sum_exp(exponents);
    }
    log_z[length] = value;
  }
  for (std::size_t particles = 1; particles <= count; ++particles) {
    std::vector<double> terms(particles);
    for (std::size_t length = 1; length <= particles; ++length) {
      terms[length - 1] = log_z[length] + log_Z[particles - length];
    }
    log_Z[particles] = log_sum_exp(terms) - std::log(static_cast<double>(particles));
  }
  return log_Z[count];
}

double twist_free_energy_curvature(const Model &model, const FreeBosonTable &table,
                                   const std::size_t axis) {
  model.validate();
  if (model.beta <= 0.0) {
    throw std::invalid_argument("twist free-energy curvature requires beta > 0");
  }
  if (axis >= model.dimension) {
    throw std::out_of_range("twist axis is out of range");
  }
  validate_table(table, model.particle_count);

  std::vector<LogDerivatives> log_z_derivatives(model.particle_count + 1);
  const double inverse_size = 1.0 / static_cast<double>(model.linear_size);
  for (std::size_t length = 1; length <= model.particle_count; ++length) {
    const double duration = static_cast<double>(length) * model.beta;
    const double scale = 2.0 * model.hopping * duration;
    double normalization = 0.0;
    double sine_sum = 0.0;
    double sine_square_sum = 0.0;
    double cosine_sum = 0.0;
    for (Coord momentum = 0; momentum < model.linear_size; ++momentum) {
      const double angle = 2.0 * std::numbers::pi * static_cast<double>(momentum) /
                           static_cast<double>(model.linear_size);
      const double weight = std::exp(scale * (std::cos(angle) - 1.0));
      normalization += weight;
      sine_sum += weight * std::sin(angle);
      sine_square_sum += weight * std::sin(angle) * std::sin(angle);
      cosine_sum += weight * std::cos(angle);
    }
    const double mean_sine = sine_sum / normalization;
    const double mean_sine_square = sine_square_sum / normalization;
    const double mean_cosine = cosine_sum / normalization;
    log_z_derivatives[length].first = -scale * inverse_size * mean_sine;
    log_z_derivatives[length].second =
        inverse_size * inverse_size *
        (-scale * mean_cosine + scale * scale * (mean_sine_square - (mean_sine * mean_sine)));
  }

  std::vector<double> first(model.particle_count + 1);
  std::vector<double> second(model.particle_count + 1);
  for (std::size_t particles = 1; particles <= model.particle_count; ++particles) {
    double probability_sum = 0.0;
    double next_first = 0.0;
    double raw_second = 0.0;
    std::vector<double> probabilities(particles);
    for (std::size_t length = 1; length <= particles; ++length) {
      probabilities[length - 1] =
          std::exp(table.log_z[length] + table.log_Z[particles - length] -
                   std::log(static_cast<double>(particles)) - table.log_Z[particles]);
      probability_sum += probabilities[length - 1];
    }
    for (std::size_t length = 1; length <= particles; ++length) {
      const double probability = probabilities[length - 1] / probability_sum;
      const double candidate_first = log_z_derivatives[length].first + first[particles - length];
      const double candidate_second = log_z_derivatives[length].second + second[particles - length];
      next_first += probability * candidate_first;
      raw_second += probability * (candidate_second + (candidate_first * candidate_first));
    }
    first[particles] = next_first;
    second[particles] = raw_second - (next_first * next_first);
  }
  const double curvature = -second[model.particle_count] / model.beta;
  return clamp_nonnegative_roundoff(curvature, std::abs(curvature));
}

EqualTimeObservables equal_time_observables(const IdealBosonConfiguration &configuration) {
  configuration.validate();
  const Model &model = configuration.model;
  const auto volume = model.volume();
  const auto time_points = configuration.time_links_per_beta;
  const auto positions = retained_positions(configuration);

  EqualTimeObservables result{
      .site_density = std::vector<double>(volume),
      .pair_correlation = std::vector<double>(volume),
      .static_structure_factor = std::vector<double>(volume),
      .onsite_occupation_probability = std::vector<double>(model.particle_count + 1),
  };
  std::vector<double> pair_counts(volume);
  std::vector<std::vector<std::size_t>> occupancies(time_points, std::vector<std::size_t>(volume));

  for (std::size_t time = 0; time < time_points; ++time) {
    for (const std::size_t site : positions[time]) {
      ++occupancies[time][site];
    }
    for (std::size_t site = 0; site < volume; ++site) {
      const auto occupation = occupancies[time][site];
      result.site_density[site] += static_cast<double>(occupation);
      ++result.onsite_occupation_probability[occupation];
      const auto occupation_value = static_cast<double>(occupation);
      result.mean_occupation_squared += occupation_value * occupation_value;
      result.mean_factorial_occupation += occupation_value * std::max(0.0, occupation_value - 1.0);
    }
    accumulate_pair_counts(positions[time], model, pair_counts);
    accumulate_structure_factor(positions[time], model, result.static_structure_factor);
  }

  const auto slice_count = static_cast<double>(time_points);
  const double site_observation_count = slice_count * static_cast<double>(volume);
  for (double &density : result.site_density) {
    density /= slice_count;
  }
  for (double &probability : result.onsite_occupation_probability) {
    probability /= site_observation_count;
  }
  result.mean_occupation_squared /= site_observation_count;
  result.mean_factorial_occupation /= site_observation_count;
  for (double &structure_factor : result.static_structure_factor) {
    structure_factor /= slice_count;
  }

  if (model.particle_count > 0) {
    const double density = static_cast<double>(model.particle_count) / static_cast<double>(volume);
    for (std::size_t displacement = 0; displacement < volume; ++displacement) {
      const double correlation = pair_counts[displacement] / site_observation_count;
      const double self_term = displacement == 0 ? density : 0.0;
      result.pair_correlation[displacement] = (correlation - self_term) / (density * density);
    }
  }
  return result;
}

ImaginaryTimeDensityCorrelations
retained_density_correlations(const IdealBosonConfiguration &configuration) {
  configuration.validate();
  const Model &model = configuration.model;
  const auto volume = model.volume();
  const auto time_points = configuration.time_links_per_beta;
  const auto positions = retained_positions(configuration);
  ImaginaryTimeDensityCorrelations result{
      .time_points = time_points,
      .spatial_points = volume,
      .connected_density = std::vector<double>(time_points * volume),
  };

  for (std::size_t lag = 0; lag < time_points; ++lag) {
    for (std::size_t origin_time = 0; origin_time < time_points; ++origin_time) {
      const std::size_t target_time = (origin_time + lag) % time_points;
      for (const std::size_t origin : positions[origin_time]) {
        for (const std::size_t target : positions[target_time]) {
          ++result.connected_density[(lag * volume) + displacement_index(origin, target, model)];
        }
      }
    }
  }

  const double normalization = static_cast<double>(time_points) * static_cast<double>(volume);
  const double density = static_cast<double>(model.particle_count) / static_cast<double>(volume);
  for (double &correlation : result.connected_density) {
    correlation = (correlation / normalization) - (density * density);
  }
  return result;
}

MatsubaraDensityCorrelations
retained_grid_matsubara_transform(const Model &model,
                                  const ImaginaryTimeDensityCorrelations &correlations) {
  model.validate();
  if (model.beta <= 0.0) {
    throw std::invalid_argument("Matsubara transform requires beta > 0");
  }
  const auto volume = model.volume();
  if (correlations.time_points == 0 || correlations.spatial_points != volume ||
      correlations.connected_density.size() != correlations.time_points * volume) {
    throw std::invalid_argument("density-correlation grid does not match the model");
  }

  const auto time_points = correlations.time_points;
  std::vector<std::complex<double>> spatial_transform(time_points * volume);
  for (std::size_t time = 0; time < time_points; ++time) {
    for (std::size_t momentum = 0; momentum < volume; ++momentum) {
      const auto momentum_components = decode_flat_index(momentum, model);
      std::complex<double> value{0.0, 0.0};
      for (std::size_t displacement = 0; displacement < volume; ++displacement) {
        const auto displacement_components = decode_flat_index(displacement, model);
        const double phase = phase_for_indices(momentum_components, displacement_components, model);
        value += correlations.connected_density[(time * volume) + displacement] *
                 std::polar(1.0, -phase);
      }
      spatial_transform[(time * volume) + momentum] = value;
    }
  }

  MatsubaraDensityCorrelations result{
      .frequencies = std::vector<double>(time_points),
      .momentum_points = volume,
      .values = std::vector<std::complex<double>>(time_points * volume),
  };
  const double time_step = model.beta / static_cast<double>(time_points);
  for (std::size_t frequency = 0; frequency < time_points; ++frequency) {
    result.frequencies[frequency] =
        2.0 * std::numbers::pi * static_cast<double>(frequency) / model.beta;
    for (std::size_t momentum = 0; momentum < volume; ++momentum) {
      std::complex<double> value{0.0, 0.0};
      for (std::size_t time = 0; time < time_points; ++time) {
        const double phase = 2.0 * std::numbers::pi * static_cast<double>(frequency * time) /
                             static_cast<double>(time_points);
        value += spatial_transform[(time * volume) + momentum] * std::polar(1.0, phase);
      }
      result.values[(frequency * volume) + momentum] = time_step * value;
    }
  }
  return result;
}

RetainedGeometryObservables
retained_geometry_observables(const IdealBosonConfiguration &configuration) {
  configuration.validate();
  const Model &model = configuration.model;
  const auto volume = model.volume();
  const auto time_points = configuration.time_links_per_beta;
  RetainedGeometryObservables result{
      .time_points = time_points,
      .displacement_points = volume,
      .mean_square_displacement = std::vector<double>(time_points),
      .return_probability = std::vector<double>(time_points),
      .displacement_probability = std::vector<double>(time_points * volume),
  };
  if (model.particle_count == 0) {
    return result;
  }

  const auto particle_count = static_cast<double>(model.particle_count);
  for (std::size_t time = 0; time < time_points; ++time) {
    for (std::size_t particle = 0; particle < model.particle_count; ++particle) {
      std::vector<std::size_t> torus_displacement(model.dimension);
      double squared_displacement = 0.0;
      for (std::size_t axis = 0; axis < model.dimension; ++axis) {
        const auto label = static_cast<ParticleId>(particle);
        const double value =
            static_cast<double>(configuration.worldlines_covering.at(label, time, axis)) -
            static_cast<double>(configuration.worldlines_covering.at(label, 0, axis));
        squared_displacement += value * value;
        const Coord torus_delta = configuration.worldlines.at(label, time, axis) -
                                  configuration.worldlines.at(label, 0, axis);
        torus_displacement[axis] =
            static_cast<std::size_t>(torus_mod(torus_delta, model.linear_size));
      }
      result.mean_square_displacement[time] += squared_displacement / particle_count;
      const auto flat = encode_components(torus_displacement, model);
      result.displacement_probability[(time * volume) + flat] += 1.0 / particle_count;
      if (flat == 0) {
        result.return_probability[time] += 1.0 / particle_count;
      }
    }
  }
  return result;
}

std::vector<RetainedCycleGeometry>
retained_cycle_geometry(const IdealBosonConfiguration &configuration) {
  configuration.validate();
  std::vector<RetainedCycleGeometry> result;
  result.reserve(configuration.cycles.size());
  for (const IdealCyclePath &cycle : configuration.cycles) {
    const auto distinct_points = cycle.labels.size() * configuration.time_links_per_beta;
    if (distinct_points == 0 || cycle.covering_path.size() != distinct_points + 1) {
      throw std::logic_error("cycle path has an invalid retained-point count");
    }
    std::vector<double> center(configuration.model.dimension);
    for (std::size_t point = 0; point < distinct_points; ++point) {
      for (std::size_t axis = 0; axis < configuration.model.dimension; ++axis) {
        center[axis] += static_cast<double>(cycle.covering_path[point][axis]);
      }
    }
    for (double &component : center) {
      component /= static_cast<double>(distinct_points);
    }

    RetainedCycleGeometry geometry{
        .length = cycle.labels.size(),
        .winding = cycle.winding,
    };
    for (std::size_t point = 0; point < distinct_points; ++point) {
      double squared_radius = 0.0;
      for (std::size_t axis = 0; axis < configuration.model.dimension; ++axis) {
        const double displacement =
            static_cast<double>(cycle.covering_path[point][axis]) - center[axis];
        squared_radius += displacement * displacement;
      }
      geometry.radius_of_gyration_squared += squared_radius / static_cast<double>(distinct_points);
      geometry.maximum_radius_squared = std::max(geometry.maximum_radius_squared, squared_radius);
    }
    result.push_back(std::move(geometry));
  }
  return result;
}

} // namespace qmc
