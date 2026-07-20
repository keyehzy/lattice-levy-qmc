#include "qmc/observables.hpp"

#include "checked_math.hpp"
#include "qmc/free_numerics.hpp"
#include "qmc/torus_layout.hpp"

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

std::vector<std::vector<SiteId>> retained_positions(const IdealBosonConfiguration &configuration,
                                                    const TorusLayout &layout) {
  const auto time_points = configuration.time_links_per_beta();
  const Model &model = configuration.model();
  const DenseWorldlines &worldlines = configuration.covering_worldlines();
  std::vector<std::vector<SiteId>> positions(time_points);
  for (std::size_t time = 0; time < time_points; ++time) {
    positions[time].reserve(model.particle_count);
    for (std::size_t particle = 0; particle < model.particle_count; ++particle) {
      positions[time].push_back(
          layout.encode_covering(worldlines.site(static_cast<ParticleId>(particle), time)));
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
                         const TorusLayout &layout) {
  double phase = 0.0;
  for (std::size_t axis = 0; axis < layout.dimension(); ++axis) {
    phase += 2.0 * std::numbers::pi * static_cast<double>(left[axis]) *
             static_cast<double>(right[axis]) / static_cast<double>(layout.linear_size());
  }
  return phase;
}

std::pair<double, double> occupation_moments(const double energy,
                                             const CanonicalEnsemble &ensemble) {
  const Model &model = ensemble.model();
  const auto log_Z = ensemble.log_partitions();
  if (model.particle_count == 0) {
    return {0.0, 0.0};
  }
  std::vector<double> occupation_terms(model.particle_count);
  for (std::size_t length = 1; length <= model.particle_count; ++length) {
    const double duration = static_cast<double>(length) * model.beta;
    occupation_terms[length - 1] =
        log_Z[model.particle_count - length] - log_Z[model.particle_count] - (duration * energy);
  }
  const double occupation = std::exp(log_sum_exp(occupation_terms));

  double factorial_moment = 0.0;
  if (model.particle_count >= 2) {
    std::vector<double> factorial_terms(model.particle_count - 1);
    for (std::size_t length = 2; length <= model.particle_count; ++length) {
      const double duration = static_cast<double>(length) * model.beta;
      factorial_terms[length - 2] = std::log(2.0 * static_cast<double>(length - 1)) +
                                    log_Z[model.particle_count - length] -
                                    log_Z[model.particle_count] - (duration * energy);
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

void accumulate_pair_counts(const std::span<const SiteId> positions, const TorusLayout &layout,
                            std::vector<double> &pair_counts) {
  for (const SiteId origin : positions) {
    for (const SiteId target : positions) {
      ++pair_counts[layout.flat_displacement(origin, target).value()];
    }
  }
}

void accumulate_structure_factor(const std::span<const SiteId> positions, const TorusLayout &layout,
                                 const Model &model, std::vector<double> &structure_factor) {
  if (model.particle_count == 0) {
    return;
  }
  std::vector<std::size_t> site_components(layout.dimension());
  for (std::size_t momentum = 0; momentum < structure_factor.size(); ++momentum) {
    const auto momentum_components = layout.decode(SiteId(momentum));
    std::complex<double> density_mode{0.0, 0.0};
    for (const SiteId site : positions) {
      layout.decode_into(site, site_components);
      const double phase = phase_for_indices(momentum_components, site_components, layout);
      density_mode += std::polar(1.0, -phase);
    }
    structure_factor[momentum] +=
        std::norm(density_mode) / static_cast<double>(model.particle_count);
  }
}

} // namespace

CanonicalThermodynamics canonical_thermodynamics(const CanonicalEnsemble &ensemble) {
  const Model &model = ensemble.model();
  const auto log_z = ensemble.log_cycle_weights();
  const auto log_Z = ensemble.log_partitions();
  if (model.beta <= 0.0) {
    throw std::invalid_argument("canonical thermodynamics requires beta > 0");
  }

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
      const double log_probability = log_z[length] + log_Z[particles - length] -
                                     std::log(static_cast<double>(particles)) - log_Z[particles];
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
    result.free_energy[particles] = -log_Z[particles] / model.beta;
    result.energy[particles] = -log_Z_first[particles];
    result.heat_capacity[particles] = model.beta * model.beta * log_Z_second[particles];
    result.entropy[particles] = log_Z[particles] + (model.beta * result.energy[particles]);
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
  const TorusLayout layout(model.linear_size, model.dimension);
  const auto volume = layout.volume();
  MomentumDistribution result;
  result.modes.reserve(volume);

  for (std::size_t flat = 0; flat < volume; ++flat) {
    MomentumMode mode;
    mode.indices = layout.decode(SiteId(flat));
    mode.wavevector.resize(model.dimension);
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      mode.wavevector[axis] = 2.0 * std::numbers::pi * static_cast<double>(mode.indices[axis]) /
                              static_cast<double>(model.linear_size);
    }
    mode.energy = mode_energy(mode.indices, model);

    const auto [occupation, variance] = occupation_moments(mode.energy, ensemble);
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

MomentumDistribution momentum_distribution(const Model &model) {
  return momentum_distribution(CanonicalEnsemble(model));
}

std::vector<OneBodyDensityPoint> one_body_density_matrix(const CanonicalEnsemble &ensemble) {
  const Model &model = ensemble.model();
  const auto log_z = ensemble.log_cycle_weights();
  const auto log_Z = ensemble.log_partitions();
  const TorusLayout layout(model.linear_size, model.dimension);
  const auto volume = layout.volume();
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
    cycle_weights[length] = std::exp(log_z[length] + log_Z[model.particle_count - length] -
                                     log_Z[model.particle_count]);
  }

  std::vector<OneBodyDensityPoint> result;
  result.reserve(volume);
  for (std::size_t flat = 0; flat < volume; ++flat) {
    const auto components = layout.decode(SiteId(flat));
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

std::vector<OneBodyDensityPoint> one_body_density_matrix(const Model &model) {
  return one_body_density_matrix(CanonicalEnsemble(model));
}

ExactCycleStatistics exact_cycle_statistics(const CanonicalEnsemble &ensemble) {
  return exact_cycle_statistics(ensemble, ensemble.model().particle_count);
}

ExactCycleStatistics exact_cycle_statistics(const CanonicalEnsemble &ensemble,
                                            const std::size_t particle_count) {
  if (particle_count > ensemble.model().particle_count) {
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

std::vector<std::size_t> sampled_cycle_histogram(const IdealBosonConfiguration &configuration) {
  std::vector<std::size_t> result(configuration.model().particle_count + 1);
  for (const Cycle &cycle : configuration.topology().cycles()) {
    ++result[cycle.size()];
  }
  return result;
}

std::size_t longest_cycle_length(const IdealBosonConfiguration &configuration) {
  std::size_t result = 0;
  for (const Cycle &cycle : configuration.topology().cycles()) {
    result = std::max(result, cycle.size());
  }
  return result;
}

Site total_winding(const IdealBosonConfiguration &configuration) {
  Site result(configuration.model().dimension);
  for (std::size_t cycle = 0; cycle < configuration.topology().cycles().size(); ++cycle) {
    const Site winding = configuration.cycle_winding(cycle);
    for (std::size_t axis = 0; axis < configuration.model().dimension; ++axis) {
      const Coord value = winding[axis];
      if ((value > 0 && result[axis] > std::numeric_limits<Coord>::max() - value) ||
          (value < 0 && result[axis] < std::numeric_limits<Coord>::min() - value)) {
        throw std::overflow_error("total winding exceeds the coordinate range");
      }
      result[axis] += value;
    }
  }
  return result;
}

double log_canonical_partition_twisted(const CanonicalEnsemble &ensemble,
                                       std::span<const double> twist) {
  const Model &model = ensemble.model();
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

double log_canonical_partition_twisted(const Model &model, std::span<const double> twist) {
  return log_canonical_partition_twisted(CanonicalEnsemble(model), twist);
}

double twist_free_energy_curvature(const CanonicalEnsemble &ensemble, const std::size_t axis) {
  const Model &model = ensemble.model();
  const auto log_z = ensemble.log_cycle_weights();
  const auto log_Z = ensemble.log_partitions();
  if (model.beta <= 0.0) {
    throw std::invalid_argument("twist free-energy curvature requires beta > 0");
  }
  if (axis >= model.dimension) {
    throw std::out_of_range("twist axis is out of range");
  }

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
          std::exp(log_z[length] + log_Z[particles - length] -
                   std::log(static_cast<double>(particles)) - log_Z[particles]);
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

double twist_free_energy_curvature(const Model &model, const std::size_t axis) {
  return twist_free_energy_curvature(CanonicalEnsemble(model), axis);
}

RetainedGrid::RetainedGrid(const double beta, TorusLayout layout, const std::size_t time_points)
    : beta_(beta), layout_(std::move(layout)), time_points_(time_points) {
  if (!std::isfinite(beta_) || beta_ < 0.0) {
    throw std::invalid_argument("retained-grid beta must be finite and nonnegative");
  }
  if (time_points_ == 0) {
    throw std::invalid_argument("retained-grid time point count must be positive");
  }
}

ImaginaryTimeDensityCorrelations::ImaginaryTimeDensityCorrelations(
    RetainedGrid grid, std::vector<double> connected_density)
    : grid_(std::move(grid)), connected_density_(std::move(connected_density)) {
  const auto expected_size = detail::checked_product(time_points(), spatial_points(),
                                                     "density-correlation grid exceeds size_t");
  if (connected_density_.size() != expected_size) {
    throw std::invalid_argument("density-correlation storage does not match its retained grid");
  }
}

double ImaginaryTimeDensityCorrelations::at(const std::size_t time_index,
                                            const SiteId displacement) const {
  if (time_index >= time_points() || displacement.value() >= spatial_points()) {
    throw std::out_of_range("density-correlation index is outside the retained grid");
  }
  return connected_density_[(time_index * spatial_points()) + displacement.value()];
}

EqualTimeObservables equal_time_observables(const IdealBosonConfiguration &configuration) {
  const Model &model = configuration.model();
  const TorusLayout layout(model.linear_size, model.dimension);
  const auto volume = layout.volume();
  const auto time_points = configuration.time_links_per_beta();
  const auto positions = retained_positions(configuration, layout);

  EqualTimeObservables result{
      .site_density = std::vector<double>(volume),
      .pair_correlation = std::vector<double>(volume),
      .static_structure_factor = std::vector<double>(volume),
      .onsite_occupation_probability = std::vector<double>(model.particle_count + 1),
  };
  std::vector<double> pair_counts(volume);
  std::vector<std::vector<std::size_t>> occupancies(time_points, std::vector<std::size_t>(volume));

  for (std::size_t time = 0; time < time_points; ++time) {
    for (const SiteId site : positions[time]) {
      ++occupancies[time][site.value()];
    }
    for (std::size_t site = 0; site < volume; ++site) {
      const auto occupation = occupancies[time][site];
      result.site_density[site] += static_cast<double>(occupation);
      ++result.onsite_occupation_probability[occupation];
      const auto occupation_value = static_cast<double>(occupation);
      result.mean_occupation_squared += occupation_value * occupation_value;
      result.mean_factorial_occupation += occupation_value * std::max(0.0, occupation_value - 1.0);
    }
    accumulate_pair_counts(positions[time], layout, pair_counts);
    accumulate_structure_factor(positions[time], layout, model, result.static_structure_factor);
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
  const Model &model = configuration.model();
  const TorusLayout layout(model.linear_size, model.dimension);
  const auto volume = layout.volume();
  const auto time_points = configuration.time_links_per_beta();
  const auto grid_size =
      detail::checked_product(time_points, volume, "density-correlation grid exceeds size_t");
  const auto positions = retained_positions(configuration, layout);
  std::vector<double> connected_density(grid_size);

  for (std::size_t lag = 0; lag < time_points; ++lag) {
    for (std::size_t origin_time = 0; origin_time < time_points; ++origin_time) {
      const std::size_t target_time = (origin_time + lag) % time_points;
      for (const SiteId origin : positions[origin_time]) {
        for (const SiteId target : positions[target_time]) {
          const auto displacement = layout.flat_displacement(origin, target);
          ++connected_density[(lag * volume) + displacement.value()];
        }
      }
    }
  }

  const double normalization = static_cast<double>(time_points) * static_cast<double>(volume);
  const double density = static_cast<double>(model.particle_count) / static_cast<double>(volume);
  for (double &correlation : connected_density) {
    correlation = (correlation / normalization) - (density * density);
  }
  return {RetainedGrid(model.beta, layout, time_points), std::move(connected_density)};
}

MatsubaraDensityCorrelations
retained_grid_matsubara_transform(const ImaginaryTimeDensityCorrelations &correlations) {
  const RetainedGrid &grid = correlations.grid();
  if (grid.beta() <= 0.0) {
    throw std::invalid_argument("Matsubara transform requires beta > 0");
  }
  const TorusLayout &layout = grid.layout();
  const auto volume = layout.volume();
  const auto time_points = grid.time_points();
  const auto connected_density = correlations.connected_density();
  const auto grid_size = connected_density.size();
  std::vector<std::complex<double>> spatial_transform(grid_size);
  std::vector<std::size_t> displacement_components(layout.dimension());
  for (std::size_t time = 0; time < time_points; ++time) {
    for (std::size_t momentum = 0; momentum < volume; ++momentum) {
      const auto momentum_components = layout.decode(SiteId(momentum));
      std::complex<double> value{0.0, 0.0};
      for (std::size_t displacement = 0; displacement < volume; ++displacement) {
        layout.decode_into(SiteId(displacement), displacement_components);
        const double phase =
            phase_for_indices(momentum_components, displacement_components, layout);
        value += connected_density[(time * volume) + displacement] * std::polar(1.0, -phase);
      }
      spatial_transform[(time * volume) + momentum] = value;
    }
  }

  MatsubaraDensityCorrelations result{
      .frequencies = std::vector<double>(time_points),
      .momentum_points = volume,
      .values = std::vector<std::complex<double>>(grid_size),
  };
  const double time_step = grid.beta() / static_cast<double>(time_points);
  for (std::size_t frequency = 0; frequency < time_points; ++frequency) {
    result.frequencies[frequency] =
        2.0 * std::numbers::pi * static_cast<double>(frequency) / grid.beta();
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
  const Model &model = configuration.model();
  const TorusLayout layout(model.linear_size, model.dimension);
  const auto volume = layout.volume();
  const auto time_points = configuration.time_links_per_beta();
  const DenseWorldlines &worldlines = configuration.covering_worldlines();
  const auto grid_size =
      detail::checked_product(time_points, volume, "retained-geometry grid exceeds size_t");
  RetainedGeometryObservables result{
      .time_points = time_points,
      .displacement_points = volume,
      .mean_square_displacement = std::vector<double>(time_points),
      .return_probability = std::vector<double>(time_points),
      .displacement_probability = std::vector<double>(grid_size),
  };
  if (model.particle_count == 0) {
    return result;
  }

  const auto particle_count = static_cast<double>(model.particle_count);
  for (std::size_t time = 0; time < time_points; ++time) {
    for (std::size_t particle = 0; particle < model.particle_count; ++particle) {
      double squared_displacement = 0.0;
      const auto label = static_cast<ParticleId>(particle);
      for (std::size_t axis = 0; axis < model.dimension; ++axis) {
        const double value = static_cast<double>(worldlines.at(label, time, axis)) -
                             static_cast<double>(worldlines.at(label, 0, axis));
        squared_displacement += value * value;
      }
      result.mean_square_displacement[time] += squared_displacement / particle_count;
      const SiteId flat =
          layout.flat_displacement(layout.encode_covering(worldlines.site(label, 0)),
                                   layout.encode_covering(worldlines.site(label, time)));
      result.displacement_probability[(time * volume) + flat.value()] += 1.0 / particle_count;
      if (flat.value() == 0) {
        result.return_probability[time] += 1.0 / particle_count;
      }
    }
  }
  return result;
}

std::vector<RetainedCycleGeometry>
retained_cycle_geometry(const IdealBosonConfiguration &configuration) {
  const Model &model = configuration.model();
  const auto time_links = configuration.time_links_per_beta();
  const DenseWorldlines &worldlines = configuration.covering_worldlines();
  const auto cycles = configuration.topology().cycles();
  std::vector<RetainedCycleGeometry> result;
  result.reserve(cycles.size());
  for (std::size_t cycle_index = 0; cycle_index < cycles.size(); ++cycle_index) {
    const Cycle &cycle = cycles[cycle_index];
    const auto distinct_points = detail::checked_product(
        cycle.size(), time_links, "cycle retained-point count exceeds size_t");
    std::vector<double> center(model.dimension);
    for (const ParticleId label : cycle) {
      for (std::size_t time = 0; time < time_links; ++time) {
        for (std::size_t axis = 0; axis < model.dimension; ++axis) {
          center[axis] += static_cast<double>(worldlines.at(label, time, axis));
        }
      }
    }
    for (double &component : center) {
      component /= static_cast<double>(distinct_points);
    }

    RetainedCycleGeometry geometry{
        .length = cycle.size(),
        .winding = configuration.cycle_winding(cycle_index),
    };
    for (const ParticleId label : cycle) {
      for (std::size_t time = 0; time < time_links; ++time) {
        double squared_radius = 0.0;
        for (std::size_t axis = 0; axis < model.dimension; ++axis) {
          const double displacement =
              static_cast<double>(worldlines.at(label, time, axis)) - center[axis];
          squared_radius += displacement * displacement;
        }
        geometry.radius_of_gyration_squared +=
            squared_radius / static_cast<double>(distinct_points);
        geometry.maximum_radius_squared = std::max(geometry.maximum_radius_squared, squared_radius);
      }
    }
    result.push_back(std::move(geometry));
  }
  return result;
}

} // namespace qmc
