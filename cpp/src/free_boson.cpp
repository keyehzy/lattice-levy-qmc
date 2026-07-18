#include "qmc/free_boson.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace qmc {
namespace {

std::uint64_t checked_bessel_order(const Coord linear_size, const std::size_t winding) {
  const auto size = static_cast<std::uint64_t>(linear_size);
  if (winding > std::numeric_limits<std::uint64_t>::max() / size) {
    throw std::overflow_error("winding Bessel order exceeds uint64 range");
  }
  return size * static_cast<std::uint64_t>(winding);
}

std::size_t grow_winding_support(const std::size_t current) {
  if (current > std::numeric_limits<std::size_t>::max() - 8) {
    throw std::overflow_error("winding support size overflowed");
  }
  const auto additive = current + 8;
  const double scaled = (1.5 * static_cast<double>(current)) + 1.0;
  if (!std::isfinite(scaled) ||
      scaled > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("winding support size overflowed");
  }
  return std::max(additive, static_cast<std::size_t>(scaled));
}

std::vector<double> evaluate_winding_weights(const Coord linear_size, const double argument,
                                             const std::size_t support) {
  std::vector<double> weights(support + 1);
  for (std::size_t winding = 0; winding <= support; ++winding) {
    weights[winding] =
        scaled_modified_bessel_i(checked_bessel_order(linear_size, winding), argument);
  }
  return weights;
}

double included_winding_weight(const std::vector<double> &weights) {
  double included = weights.front();
  for (std::size_t winding = 1; winding < weights.size(); ++winding) {
    included += 2.0 * weights[winding];
  }
  if (!std::isfinite(included) || included <= 0.0) {
    throw std::runtime_error("failed to evaluate winding weights");
  }
  return included;
}

bool winding_tail_is_controlled(const Coord linear_size, const double argument,
                                const std::size_t support, const double included,
                                const double log_tolerance) {
  if (support > std::numeric_limits<std::size_t>::max() - 2) {
    throw std::overflow_error("winding tail support overflowed");
  }
  const double first_omitted =
      scaled_modified_bessel_i(checked_bessel_order(linear_size, support + 1), argument);
  if (first_omitted == 0.0) {
    return true;
  }
  const double second_omitted =
      scaled_modified_bessel_i(checked_bessel_order(linear_size, support + 2), argument);
  const double ratio = second_omitted / first_omitted;
  if (ratio >= 1.0) {
    return false;
  }
  const double tail_bound = 2.0 * first_omitted / (1.0 - ratio);
  return std::log(tail_bound) - std::log(included) <= log_tolerance;
}

std::vector<double> find_winding_support(const Coord linear_size, const double argument,
                                         const NumericalOptions &options, std::size_t &support) {
  const double log_tolerance = std::log(options.tail_tolerance);
  while (true) {
    if (support > options.max_winding) {
      throw std::runtime_error("winding support exceeded max_winding");
    }
    auto weights = evaluate_winding_weights(linear_size, argument, support);
    const double included = included_winding_weight(weights);
    if (winding_tail_is_controlled(linear_size, argument, support, included, log_tolerance)) {
      return weights;
    }
    support = grow_winding_support(support);
  }
}

} // namespace

double log_one_particle_trace(const double duration, const Coord linear_size,
                              const std::size_t dimension, const double hopping) {
  if (!std::isfinite(duration) || duration < 0.0) {
    throw std::invalid_argument("duration must be finite and nonnegative");
  }
  if (linear_size < 1 || dimension < 1) {
    throw std::invalid_argument("linear_size and dimension must be positive");
  }
  if (!std::isfinite(hopping) || hopping < 0.0) {
    throw std::invalid_argument("hopping must be finite and nonnegative");
  }
  const double scale = 2.0 * hopping * duration;
  if (!std::isfinite(scale)) {
    throw std::overflow_error("one-particle trace exponent scale overflowed");
  }

  std::vector<double> exponents(static_cast<std::size_t>(linear_size));
  for (Coord momentum = 0; momentum < linear_size; ++momentum) {
    const double angle =
        2.0 * std::numbers::pi * static_cast<double>(momentum) / static_cast<double>(linear_size);
    exponents[static_cast<std::size_t>(momentum)] = scale * std::cos(angle);
  }
  return static_cast<double>(dimension) * log_sum_exp(exponents);
}

double log_one_particle_trace(const double duration, const Model &model) {
  model.validate();
  return log_one_particle_trace(duration, model.linear_size, model.dimension, model.hopping);
}

FreeBosonTable canonical_table(const Model &model) {
  model.validate();
  if (model.particle_count == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("canonical table size exceeds size_t");
  }

  const auto count = model.particle_count;
  const double negative_infinity = -std::numeric_limits<double>::infinity();
  FreeBosonTable table{
      .log_z = std::vector<double>(count + 1, negative_infinity),
      .log_Z = std::vector<double>(count + 1, negative_infinity),
  };
  table.log_Z[0] = 0.0;

  for (std::size_t length = 1; length <= count; ++length) {
    const double duration = static_cast<double>(length) * model.beta;
    if (!std::isfinite(duration)) {
      throw std::overflow_error("cycle duration overflowed");
    }
    table.log_z[length] = log_one_particle_trace(duration, model);
  }

  for (std::size_t particles = 1; particles <= count; ++particles) {
    std::vector<double> terms(particles);
    for (std::size_t length = 1; length <= particles; ++length) {
      terms[length - 1] = table.log_z[length] + table.log_Z[particles - length];
    }
    table.log_Z[particles] = log_sum_exp(terms) - std::log(static_cast<double>(particles));
  }
  return table;
}

std::vector<Cycle> sample_cycle_labels(const std::size_t particle_count,
                                       const FreeBosonTable &table, Random &random) {
  if (particle_count > std::numeric_limits<ParticleId>::max()) {
    throw std::invalid_argument("particle_count exceeds the ParticleId range");
  }
  if (table.log_z.size() < particle_count + 1 || table.log_Z.size() < particle_count + 1) {
    throw std::invalid_argument("canonical table is too short");
  }
  if (particle_count == 0) {
    return {};
  }

  std::vector<ParticleId> remaining(particle_count);
  for (std::size_t index = 0; index < particle_count; ++index) {
    remaining[index] = static_cast<ParticleId>(index);
  }
  std::vector<Cycle> cycles;

  while (!remaining.empty()) {
    const auto remaining_count = remaining.size();
    std::vector<double> log_probabilities(remaining_count);
    for (std::size_t length = 1; length <= remaining_count; ++length) {
      log_probabilities[length - 1] = table.log_z[length] + table.log_Z[remaining_count - length] -
                                      std::log(static_cast<double>(remaining_count)) -
                                      table.log_Z[remaining_count];
    }
    const double normalization = log_sum_exp(log_probabilities);
    std::vector<double> probabilities(remaining_count);
    for (std::size_t index = 0; index < remaining_count; ++index) {
      probabilities[index] = std::exp(log_probabilities[index] - normalization);
    }
    const std::size_t length = random.discrete_index(probabilities) + 1;

    Cycle cycle;
    cycle.reserve(length);
    cycle.push_back(remaining.front());
    if (length > 1) {
      std::vector<ParticleId> pool(remaining.begin() + 1, remaining.end());
      for (std::size_t selected = 0; selected < length - 1; ++selected) {
        const auto choices = static_cast<std::uint64_t>(pool.size() - selected);
        const auto offset = static_cast<std::size_t>(random.uniform_index(choices));
        std::swap(pool[selected], pool[selected + offset]);
        cycle.push_back(pool[selected]);
      }
    }

    std::vector<bool> chosen(particle_count, false);
    for (const ParticleId label : cycle) {
      chosen[label] = true;
    }
    std::erase_if(remaining, [&chosen](const ParticleId label) { return chosen[label]; });
    cycles.push_back(std::move(cycle));
  }
  return cycles;
}

Coord sample_winding_1d(const Coord linear_size, const double duration, const double hopping,
                        Random &random, const NumericalOptions &options) {
  if (linear_size < 1) {
    throw std::invalid_argument("linear_size must be positive");
  }
  if (!std::isfinite(duration) || duration < 0.0) {
    throw std::invalid_argument("duration must be finite and nonnegative");
  }
  if (!std::isfinite(hopping) || hopping < 0.0) {
    throw std::invalid_argument("hopping must be finite and nonnegative");
  }
  options.validate();

  const double argument = 2.0 * hopping * duration;
  if (!std::isfinite(argument)) {
    throw std::overflow_error("winding Bessel argument overflowed");
  }
  if (argument == 0.0) {
    return 0;
  }

  const double initial =
      std::ceil(8.0 * std::sqrt(std::max(argument, 1.0)) / static_cast<double>(linear_size)) + 4.0;
  if (!std::isfinite(initial) ||
      initial > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("initial winding support overflowed");
  }
  std::size_t support = std::max<std::size_t>(4, static_cast<std::size_t>(initial));
  const std::vector<double> nonnegative_weights =
      find_winding_support(linear_size, argument, options, support);

  if (support > static_cast<std::size_t>(std::numeric_limits<Coord>::max()) ||
      support > (std::numeric_limits<std::size_t>::max() - 1) / 2) {
    throw std::overflow_error("signed winding support cannot be represented");
  }
  std::vector<double> signed_weights((2 * support) + 1);
  for (std::size_t index = 0; index < signed_weights.size(); ++index) {
    const auto distance = index > support ? index - support : support - index;
    signed_weights[index] = nonnegative_weights[distance];
  }
  const auto selected = random.discrete_index(signed_weights);
  if (selected >= support) {
    return static_cast<Coord>(selected - support);
  }
  return -static_cast<Coord>(support - selected);
}

} // namespace qmc
