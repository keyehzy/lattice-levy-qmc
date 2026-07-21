#include "qmc/free_boson.hpp"

#include "adaptive_discrete_support.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <utility>

namespace qmc {
namespace {

std::uint64_t checked_bessel_order(const Coord linear_size, const std::size_t winding) {
  const auto size = static_cast<std::uint64_t>(linear_size);
  if (winding > std::numeric_limits<std::uint64_t>::max() / size) {
    throw std::overflow_error("winding Bessel order exceeds uint64 range");
  }
  return size * static_cast<std::uint64_t>(winding);
}

std::optional<double> winding_log_tail_bound(const Coord linear_size, const double argument,
                                             const std::size_t support) {
  if (support > std::numeric_limits<std::size_t>::max() - 2) {
    throw std::overflow_error("winding tail support overflowed");
  }
  const double first_omitted =
      scaled_modified_bessel_i(checked_bessel_order(linear_size, support + 1), argument);
  if (first_omitted == 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  const double second_omitted =
      scaled_modified_bessel_i(checked_bessel_order(linear_size, support + 2), argument);
  const double ratio = second_omitted / first_omitted;
  if (!std::isfinite(ratio) || ratio >= 1.0) {
    return std::nullopt;
  }
  const double tail_bound = 2.0 * first_omitted / (1.0 - ratio);
  return std::log(tail_bound);
}

struct SymmetricWindingWeights {
  std::size_t support;
  std::vector<double> nonnegative;
};

SymmetricWindingWeights find_winding_support(const Coord linear_size, const double argument,
                                             const NumericalOptions &options,
                                             const std::size_t initial_support) {
  detail::AdaptiveDiscreteSupport support(
      initial_support, options.max_winding, options.tail_tolerance,
      "winding support exceeded max_winding", "winding support size overflowed",
      "failed to evaluate winding weights");
  std::vector<double> weights;
  while (true) {
    if (support.support() == std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("winding support size overflowed");
    }
    const std::size_t required_size = support.support() + 1;
    weights.reserve(required_size);
    for (std::size_t winding = weights.size(); winding < required_size; ++winding) {
      const double weight =
          scaled_modified_bessel_i(checked_bessel_order(linear_size, winding), argument);
      weights.push_back(weight);
      support.add_weight(weight, winding == 0 ? 1.0 : 2.0);
    }
    if (support.tail_is_controlled(
            winding_log_tail_bound(linear_size, argument, support.support()))) {
      return SymmetricWindingWeights{
          .support = support.support(),
          .nonnegative = std::move(weights),
      };
    }
    support.grow(detail::SupportGrowth{.minimum = 8, .factor = 1.5});
  }
}

} // namespace

OneParticleSpectrum::OneParticleSpectrum(const Coord linear_size, const std::size_t dimension,
                                         const double hopping)
    : layout_(linear_size, dimension), hopping_(hopping) {
  if (!std::isfinite(hopping_) || hopping_ < 0.0) {
    throw std::invalid_argument("hopping must be finite and nonnegative");
  }

  const auto size = static_cast<std::size_t>(layout_.linear_size());
  wavevectors_.reserve(size);
  cosines_.reserve(size);
  sines_.reserve(size);
  for (std::size_t momentum = 0; momentum < size; ++momentum) {
    const double wavevector = 2.0 * std::numbers::pi * static_cast<double>(momentum) /
                              static_cast<double>(layout_.linear_size());
    wavevectors_.push_back(wavevector);
    cosines_.push_back(std::cos(wavevector));
    sines_.push_back(std::sin(wavevector));
  }
}

OneParticleSpectrum::OneParticleSpectrum(const Model &model)
    : OneParticleSpectrum(model.linear_size(), model.dimension(), model.hopping()) {}

double OneParticleSpectrum::energy(const std::span<const std::size_t> momentum_components) const {
  if (momentum_components.size() != layout_.dimension()) {
    throw std::invalid_argument("momentum component count does not match the lattice dimension");
  }
  double cosine_sum = 0.0;
  for (const std::size_t component : momentum_components) {
    if (component >= cosines_.size()) {
      throw std::out_of_range("momentum component is outside [0, L)");
    }
    cosine_sum += cosines_[component];
  }
  const double result = -2.0 * hopping_ * cosine_sum;
  if (!std::isfinite(result)) {
    throw std::overflow_error("one-particle energy overflowed");
  }
  return result;
}

namespace {

double log_one_particle_trace(const double duration, const OneParticleSpectrum &spectrum,
                              std::vector<double> &exponents) {
  if (!std::isfinite(duration) || duration < 0.0) {
    throw std::invalid_argument("duration must be finite and nonnegative");
  }
  const double scale = 2.0 * spectrum.hopping() * duration;
  if (!std::isfinite(scale)) {
    throw std::overflow_error("one-particle trace exponent scale overflowed");
  }

  const auto cosines = spectrum.cosines();
  if (exponents.size() != cosines.size()) {
    throw std::logic_error("one-particle trace scratch has the wrong size");
  }
  for (std::size_t momentum = 0; momentum < cosines.size(); ++momentum) {
    exponents[momentum] = scale * cosines[momentum];
  }
  return static_cast<double>(spectrum.layout().dimension()) * log_sum_exp(exponents);
}

} // namespace

double log_one_particle_trace(const double duration, const OneParticleSpectrum &spectrum) {
  std::vector<double> exponents(spectrum.cosines().size());
  return log_one_particle_trace(duration, spectrum, exponents);
}

double log_one_particle_trace(const double duration, const Coord linear_size,
                              const std::size_t dimension, const double hopping) {
  if (!std::isfinite(duration) || duration < 0.0) {
    throw std::invalid_argument("duration must be finite and nonnegative");
  }
  return log_one_particle_trace(duration, OneParticleSpectrum(linear_size, dimension, hopping));
}

double log_one_particle_trace(const double duration, const Model &model) {
  return log_one_particle_trace(duration, OneParticleSpectrum(model));
}

CanonicalEnsemble::CanonicalEnsemble(Model model) : model_(model), spectrum_(model_) {
  if (model_.particle_count() == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("canonical table size exceeds size_t");
  }

  const auto count = model_.particle_count();
  const double negative_infinity = -std::numeric_limits<double>::infinity();
  log_z_.assign(count + 1, negative_infinity);
  log_Z_.assign(count + 1, negative_infinity);
  log_Z_[0] = 0.0;
  std::vector<double> trace_exponents(spectrum_.cosines().size());

  for (std::size_t length = 1; length <= count; ++length) {
    const double duration = static_cast<double>(length) * model_.beta();
    if (!std::isfinite(duration)) {
      throw std::overflow_error("cycle duration overflowed");
    }
    log_z_[length] = log_one_particle_trace(duration, spectrum_, trace_exponents);
    if (!std::isfinite(log_z_[length])) {
      throw std::overflow_error("canonical one-particle trace is non-finite");
    }
  }

  for (std::size_t particles = 1; particles <= count; ++particles) {
    std::vector<double> terms(particles);
    for (std::size_t length = 1; length <= particles; ++length) {
      terms[length - 1] = log_z_[length] + log_Z_[particles - length];
    }
    log_Z_[particles] = log_sum_exp(terms) - std::log(static_cast<double>(particles));
    if (!std::isfinite(log_Z_[particles])) {
      throw std::overflow_error("canonical partition recursion is non-finite");
    }
  }
}

double CanonicalEnsemble::log_partition(const std::size_t particle_count) const {
  if (particle_count > model_.particle_count()) {
    throw std::out_of_range("canonical particle count exceeds the ensemble capacity");
  }
  return log_Z_[particle_count];
}

std::vector<Cycle> CanonicalEnsemble::sample_cycles(Random &random) const {
  return sample_cycles(model_.particle_count(), random);
}

std::vector<Cycle> CanonicalEnsemble::sample_cycles(const std::size_t particle_count,
                                                    Random &random) const {
  if (particle_count > model_.particle_count()) {
    throw std::out_of_range("cycle-sampling particle count exceeds the ensemble capacity");
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
    std::vector<double> log_weights(remaining_count);
    for (std::size_t length = 1; length <= remaining_count; ++length) {
      log_weights[length - 1] = log_z_[length] + log_Z_[remaining_count - length];
    }
    const std::size_t length = random.discrete_log_index(log_weights) + 1;

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
  const SymmetricWindingWeights winding_weights = find_winding_support(
      linear_size, argument, options, std::max<std::size_t>(4, static_cast<std::size_t>(initial)));
  const std::size_t support = winding_weights.support;

  if (support > static_cast<std::size_t>(std::numeric_limits<Coord>::max()) ||
      support > (std::numeric_limits<std::size_t>::max() - 1) / 2) {
    throw std::overflow_error("signed winding support cannot be represented");
  }
  std::vector<double> signed_weights((2 * support) + 1);
  for (std::size_t index = 0; index < signed_weights.size(); ++index) {
    const auto distance = index > support ? index - support : support - index;
    signed_weights[index] = winding_weights.nonnegative[distance];
  }
  const auto selected = random.discrete_index(signed_weights);
  if (selected >= support) {
    return static_cast<Coord>(selected - support);
  }
  return -static_cast<Coord>(support - selected);
}

} // namespace qmc
