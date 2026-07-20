#include "qmc/random.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace qmc {

Random::Random(const std::uint64_t seed) : engine_(seed) {}

double Random::uniform_unit() {
  return std::generate_canonical<double, std::numeric_limits<double>::digits>(engine_);
}

double Random::uniform_open() {
  double value = 0.0;
  do {
    value = uniform_unit();
  } while (value == 0.0);
  return value;
}

std::uint64_t Random::uniform_index(const std::uint64_t upper_exclusive) {
  if (upper_exclusive == 0) {
    throw std::invalid_argument("upper_exclusive must be positive");
  }
  std::uniform_int_distribution<std::uint64_t> distribution(0, upper_exclusive - 1);
  return distribution(engine_);
}

std::uint64_t Random::binomial(const std::uint64_t trials, const double probability) {
  if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
    throw std::invalid_argument("binomial probability must lie in [0, 1]");
  }
  std::binomial_distribution<std::uint64_t> distribution(trials, probability);
  return distribution(engine_);
}

std::size_t Random::discrete_index(const std::span<const double> weights) {
  if (weights.empty()) {
    throw std::invalid_argument("discrete weights must not be empty");
  }

  double total = 0.0;
  for (const double weight : weights) {
    if (!std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument("discrete weights must be finite and nonnegative");
    }
    total += weight;
  }
  if (!std::isfinite(total) || total <= 0.0) {
    throw std::invalid_argument("discrete weights must have a finite positive sum");
  }

  const double draw = uniform_unit() * total;
  double cumulative = 0.0;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    cumulative += weights[index];
    if (draw < cumulative) {
      return index;
    }
  }
  return weights.size() - 1;
}

std::size_t Random::discrete_log_index(const std::span<const double> log_weights) {
  if (log_weights.empty()) {
    throw std::invalid_argument("discrete log weights must not be empty");
  }

  double maximum = -std::numeric_limits<double>::infinity();
  for (const double log_weight : log_weights) {
    if (std::isnan(log_weight) || log_weight == std::numeric_limits<double>::infinity()) {
      throw std::invalid_argument("discrete log weights must be finite or negative infinity");
    }
    maximum = std::max(maximum, log_weight);
  }
  if (maximum == -std::numeric_limits<double>::infinity()) {
    throw std::invalid_argument("discrete log weights must contain positive probability mass");
  }

  double scaled_total = 0.0;
  for (const double log_weight : log_weights) {
    scaled_total += std::exp(log_weight - maximum);
  }
  if (!std::isfinite(scaled_total) || scaled_total <= 0.0) {
    throw std::invalid_argument("discrete log weights must have a finite positive scaled sum");
  }

  const double draw = uniform_unit() * scaled_total;
  double cumulative = 0.0;
  std::size_t last_positive = 0;
  for (std::size_t index = 0; index < log_weights.size(); ++index) {
    if (log_weights[index] == -std::numeric_limits<double>::infinity()) {
      continue;
    }
    last_positive = index;
    cumulative += std::exp(log_weights[index] - maximum);
    if (draw < cumulative) {
      return index;
    }
  }
  return last_positive;
}

void Random::shuffle(const std::span<ParticleId> labels) {
  std::shuffle(labels.begin(), labels.end(), engine_);
}

} // namespace qmc
