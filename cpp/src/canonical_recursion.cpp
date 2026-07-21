#include "canonical_recursion.hpp"

#include "qmc/free_numerics.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qmc::detail {

std::vector<double> canonical_log_partitions(const std::span<const double> log_cycle_weights) {
  if (log_cycle_weights.empty()) {
    throw std::invalid_argument("canonical cycle weights must include the unused zero index");
  }

  const std::size_t particle_count = log_cycle_weights.size() - 1;
  const double negative_infinity = -std::numeric_limits<double>::infinity();
  for (const double log_cycle_weight : log_cycle_weights.subspan(1)) {
    if (std::isnan(log_cycle_weight) ||
        log_cycle_weight == std::numeric_limits<double>::infinity()) {
      throw std::overflow_error("canonical cycle weight is non-finite");
    }
  }

  std::vector<double> log_partitions(log_cycle_weights.size(), negative_infinity);
  std::vector<double> terms(particle_count);
  log_partitions[0] = 0.0;

  for (std::size_t particles = 1; particles <= particle_count; ++particles) {
    for (std::size_t length = 1; length <= particles; ++length) {
      terms[length - 1] = log_cycle_weights[length] + log_partitions[particles - length];
    }
    log_partitions[particles] = log_sum_exp(std::span<const double>(terms.data(), particles)) -
                                std::log(static_cast<double>(particles));
    if (!std::isfinite(log_partitions[particles])) {
      throw std::overflow_error("canonical partition recursion is non-finite");
    }
  }
  return log_partitions;
}

} // namespace qmc::detail
