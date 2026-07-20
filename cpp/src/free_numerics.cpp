#include "qmc/free_numerics.hpp"

#include "adaptive_discrete_support.hpp"

#include <algorithm>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>
#include <limits>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace qmc {
namespace {

void validate_nonnegative_time(const double value, const char *name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
  }
}

void validate_hopping(const double hopping) {
  if (!std::isfinite(hopping) || hopping < 0.0) {
    throw std::invalid_argument("hopping must be finite and nonnegative");
  }
}

std::uint64_t coordinate_offset(const Coord value) {
  constexpr auto minimum = std::numeric_limits<Coord>::min();
  return static_cast<std::uint64_t>(value) - static_cast<std::uint64_t>(minimum);
}

Coord coordinate_from_offset(const std::uint64_t offset) {
  constexpr auto sign_offset = std::uint64_t{1} << 63U;
  if (offset < sign_offset) {
    return std::numeric_limits<Coord>::min() + static_cast<Coord>(offset);
  }
  return static_cast<Coord>(offset - sign_offset);
}

std::pair<bool, std::uint64_t> displacement(const Coord a, const Coord b) {
  const auto a_offset = coordinate_offset(a);
  const auto b_offset = coordinate_offset(b);
  if (b_offset >= a_offset) {
    return {true, b_offset - a_offset};
  }
  return {false, a_offset - b_offset};
}

Coord apply_jump_counts(const Coord start, const std::uint64_t plus, const std::uint64_t minus) {
  auto offset = coordinate_offset(start);
  if (plus >= minus) {
    const auto increase = plus - minus;
    if (increase > std::numeric_limits<std::uint64_t>::max() - offset) {
      throw std::overflow_error("sampled covering coordinate exceeds int64 range");
    }
    offset += increase;
  } else {
    const auto decrease = minus - plus;
    if (decrease > offset) {
      throw std::overflow_error("sampled covering coordinate exceeds int64 range");
    }
    offset -= decrease;
  }
  return coordinate_from_offset(offset);
}

} // namespace

double scaled_modified_bessel_i(const std::uint64_t order, const double argument) {
  if (!std::isfinite(argument) || argument < 0.0) {
    throw std::invalid_argument("Bessel argument must be finite and nonnegative");
  }
  if (order > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Bessel order exceeds the GSL integer range");
  }

  // The _e API reports underflow through its return code, but GSL's process-
  // global default handler aborts before callers can inspect that code.
  static std::once_flag gsl_error_handler_flag;
  std::call_once(gsl_error_handler_flag, [] { gsl_set_error_handler_off(); });

  gsl_sf_result result{};
  const int status = gsl_sf_bessel_In_scaled_e(static_cast<int>(order), argument, &result);
  if (status == GSL_EUNDRFLW) {
    return 0.0;
  }
  if (status != GSL_SUCCESS) {
    throw std::runtime_error(std::string("GSL failed to evaluate scaled Bessel I: ") +
                             gsl_strerror(status));
  }
  if (!std::isfinite(result.val) || result.val < 0.0) {
    throw std::runtime_error("GSL returned an invalid scaled Bessel I value");
  }
  return result.val;
}

namespace {

std::uint64_t checked_count_sum(const std::uint64_t left, const std::uint64_t right) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error("conditioned jump count exceeds uint64 range");
  }
  return left + right;
}

} // namespace

Coord torus_mod(const Coord coordinate, const Coord linear_size) {
  if (linear_size < 1) {
    throw std::invalid_argument("linear_size must be positive");
  }
  const Coord remainder = coordinate % linear_size;
  return remainder < 0 ? remainder + linear_size : remainder;
}

double log_sum_exp(const std::span<const double> values) {
  if (values.empty()) {
    throw std::invalid_argument("log_sum_exp requires at least one value");
  }
  for (const double value : values) {
    if (std::isnan(value)) {
      throw std::invalid_argument("log_sum_exp input must not contain NaN");
    }
  }

  const double maximum = *std::ranges::max_element(values);
  if (maximum == -std::numeric_limits<double>::infinity()) {
    return maximum;
  }
  if (maximum == std::numeric_limits<double>::infinity()) {
    return maximum;
  }

  double scaled_sum = 0.0;
  for (const double value : values) {
    scaled_sum += std::exp(value - maximum);
  }
  return maximum + std::log(scaled_sum);
}

std::uint64_t sample_bessel_pair_count(const std::uint64_t abs_delta, const double lambda,
                                       Random &random, const NumericalOptions &options) {
  options.validate();
  if (!std::isfinite(lambda) || lambda < 0.0) {
    throw std::invalid_argument("lambda must be finite and nonnegative");
  }
  if (lambda == 0.0) {
    if (abs_delta != 0) {
      throw std::invalid_argument(
          "a nonzero endpoint displacement has zero weight when lambda is zero");
    }
    return 0;
  }

  const auto delta = static_cast<double>(abs_delta);
  const double mode_value = 0.5 * (std::hypot(delta, 2.0 * lambda) - delta);
  if (!std::isfinite(mode_value) || mode_value < 0.0 ||
      mode_value > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("Bessel-count mode exceeds addressable support");
  }
  const auto mode = static_cast<std::size_t>(std::floor(mode_value));
  const double width_value = std::ceil(12.0 * std::sqrt(lambda + 1.0)) + 8.0;
  if (!std::isfinite(width_value) ||
      width_value > static_cast<double>(std::numeric_limits<std::size_t>::max() - mode)) {
    throw std::overflow_error("Bessel-count initial support overflowed");
  }
  const std::size_t initial_support =
      std::max<std::size_t>(16, mode + static_cast<std::size_t>(width_value));

  const double log_lambda = std::log(lambda);
  detail::AdaptiveDiscreteSupport support(
      initial_support, options.max_bessel_terms, options.tail_tolerance,
      "Bessel-count support exceeded max_bessel_terms", "discrete support size overflowed",
      "failed to evaluate Bessel-count weights");
  std::vector<double> log_weights;

  while (true) {
    if (support.support() == std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("discrete support size overflowed");
    }
    const std::size_t required_size = support.support() + 1;
    log_weights.reserve(required_size);
    if (log_weights.empty()) {
      const double log_weight = (delta * log_lambda) - std::lgamma(delta + 1.0);
      log_weights.push_back(log_weight);
      support.add_log_weight(log_weight);
    }
    while (log_weights.size() < required_size) {
      const std::size_t index = log_weights.size();
      const auto count = static_cast<double>(index);
      const double log_weight =
          log_weights.back() + (2.0 * log_lambda) - std::log(count) - std::log(count + delta);
      log_weights.push_back(log_weight);
      support.add_log_weight(log_weight);
    }

    if (support.support() > std::numeric_limits<std::size_t>::max() - 2) {
      throw std::overflow_error("Bessel-count tail support overflowed");
    }
    const double first = static_cast<double>(support.support()) + 1.0;
    const double log_first_omitted =
        log_weights.back() + (2.0 * log_lambda) - std::log(first) - std::log(first + delta);
    const double next = static_cast<double>(support.support()) + 2.0;
    const double ratio_after_first = (lambda / next) * (lambda / (next + delta));

    if (!std::isfinite(ratio_after_first) || ratio_after_first >= 1.0) {
      support.grow(detail::SupportGrowth{.minimum = 32, .factor = 2.0});
      continue;
    }

    const double log_tail_bound = log_first_omitted - std::log1p(-ratio_after_first);
    if (support.tail_is_controlled(log_tail_bound)) {
      return static_cast<std::uint64_t>(random.discrete_log_index(log_weights));
    }
    support.grow(detail::SupportGrowth{.minimum = 32, .factor = 1.5});
  }
}

Coord sample_midpoint_covering_1d(const Coord a, const Coord b, const double tau_left,
                                  const double tau_right, const double hopping, Random &random,
                                  const NumericalOptions &options) {
  validate_nonnegative_time(tau_left, "tau_left");
  validate_nonnegative_time(tau_right, "tau_right");
  validate_hopping(hopping);
  options.validate();

  const double total = tau_left + tau_right;
  if (!std::isfinite(total)) {
    throw std::overflow_error("total bridge duration overflowed");
  }
  if (total == 0.0) {
    if (a != b) {
      throw std::invalid_argument("endpoints must coincide for a zero-duration bridge");
    }
    return a;
  }
  if (tau_left == 0.0) {
    return a;
  }
  if (tau_right == 0.0) {
    return b;
  }

  const auto [positive, abs_delta] = displacement(a, b);
  const double lambda = hopping * total;
  if (!std::isfinite(lambda)) {
    throw std::overflow_error("hopping times bridge duration overflowed");
  }
  if (lambda == 0.0) {
    if (abs_delta != 0) {
      throw std::invalid_argument("the requested bridge has zero free-particle weight");
    }
    return a;
  }

  const std::uint64_t pairs = sample_bessel_pair_count(abs_delta, lambda, random, options);
  const std::uint64_t larger = checked_count_sum(pairs, abs_delta);
  const std::uint64_t plus = positive ? larger : pairs;
  const std::uint64_t minus = positive ? pairs : larger;
  const double fraction_left = tau_left / total;
  return apply_jump_counts(a, random.binomial(plus, fraction_left),
                           random.binomial(minus, fraction_left));
}

Site sample_midpoint_covering(const Site &a, const Site &b, const double tau_left,
                              const double tau_right, const double hopping, Random &random,
                              const NumericalOptions &options) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("a and b must have equal dimensions");
  }
  Site midpoint(a.size());
  for (std::size_t axis = 0; axis < a.size(); ++axis) {
    midpoint[axis] = sample_midpoint_covering_1d(a[axis], b[axis], tau_left, tau_right, hopping,
                                                 random, options);
  }
  return midpoint;
}

CoveringPath sample_bridge_covering(const Site &a, const Site &b, const double total_time,
                                    const std::size_t steps, const double hopping, Random &random,
                                    const NumericalOptions &options) {
  validate_nonnegative_time(total_time, "total_time");
  validate_hopping(hopping);
  options.validate();
  if (steps < 1) {
    throw std::invalid_argument("steps must be at least one");
  }
  if (a.size() != b.size()) {
    throw std::invalid_argument("a and b must have equal dimensions");
  }
  if (total_time == 0.0 && a != b) {
    throw std::invalid_argument("endpoints must coincide for a zero-duration bridge");
  }
  if (steps == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("bridge point count exceeds size_t");
  }

  CoveringPath path(steps + 1, Site(a.size()));
  path.front() = a;
  path.back() = b;
  const double time_step = total_time / static_cast<double>(steps);

  std::vector<std::pair<std::size_t, std::size_t>> stack{{0, steps}};
  while (!stack.empty()) {
    const auto [left, right] = stack.back();
    stack.pop_back();
    const auto width = right - left;
    if (width <= 1) {
      continue;
    }

    const auto middle = left + (width / 2);
    const double tau_left = static_cast<double>(middle - left) * time_step;
    const double tau_right = static_cast<double>(right - middle) * time_step;
    path[middle] = sample_midpoint_covering(path[left], path[right], tau_left, tau_right, hopping,
                                            random, options);
    stack.emplace_back(middle, right);
    stack.emplace_back(left, middle);
  }
  return path;
}

double periodic_kernel_scaled_1d(const Coord delta, const double duration, const Coord linear_size,
                                 const double hopping) {
  validate_nonnegative_time(duration, "duration");
  validate_hopping(hopping);
  if (linear_size < 1) {
    throw std::invalid_argument("linear_size must be positive");
  }
  const double exponent_scale = 2.0 * hopping * duration;
  if (!std::isfinite(exponent_scale)) {
    throw std::overflow_error("kernel exponent scale overflowed");
  }

  double sum = 0.0;
  for (Coord momentum = 0; momentum < linear_size; ++momentum) {
    const double angle =
        2.0 * std::numbers::pi * static_cast<double>(momentum) / static_cast<double>(linear_size);
    sum += std::exp(exponent_scale * (std::cos(angle) - 1.0)) *
           std::cos(angle * static_cast<double>(delta));
  }
  return std::max(0.0, sum / static_cast<double>(linear_size));
}

Coord sample_midpoint_torus_1d(const Coord a, const Coord b, const double tau_left,
                               const double tau_right, const Coord linear_size,
                               const double hopping, Random &random) {
  if (linear_size < 1) {
    throw std::invalid_argument("linear_size must be positive");
  }
  const auto site_count = static_cast<std::size_t>(linear_size);
  std::vector<double> weights(site_count);
  for (std::size_t index = 0; index < site_count; ++index) {
    const auto site = static_cast<Coord>(index);
    const auto [left_positive, left_delta] = displacement(a, site);
    const auto [right_positive, right_delta] = displacement(site, b);
    if (left_delta > static_cast<std::uint64_t>(std::numeric_limits<Coord>::max()) ||
        right_delta > static_cast<std::uint64_t>(std::numeric_limits<Coord>::max())) {
      throw std::overflow_error("torus midpoint displacement exceeds int64 range");
    }
    const Coord signed_left =
        left_positive ? static_cast<Coord>(left_delta) : -static_cast<Coord>(left_delta);
    const Coord signed_right =
        right_positive ? static_cast<Coord>(right_delta) : -static_cast<Coord>(right_delta);
    weights[index] = periodic_kernel_scaled_1d(signed_left, tau_left, linear_size, hopping) *
                     periodic_kernel_scaled_1d(signed_right, tau_right, linear_size, hopping);
  }
  return static_cast<Coord>(random.discrete_index(weights));
}

std::vector<double> exact_midpoint_pmf_window(const Coord a, const Coord b, const double tau_left,
                                              const double tau_right, const double hopping,
                                              const std::span<const Coord> coordinates) {
  validate_nonnegative_time(tau_left, "tau_left");
  validate_nonnegative_time(tau_right, "tau_right");
  validate_hopping(hopping);

  const double left_argument = 2.0 * hopping * tau_left;
  const double right_argument = 2.0 * hopping * tau_right;
  const double total_argument = left_argument + right_argument;
  std::vector<double> probabilities(coordinates.size(), 0.0);
  if (total_argument == 0.0) {
    if (a == b) {
      for (std::size_t index = 0; index < coordinates.size(); ++index) {
        probabilities[index] = coordinates[index] == a ? 1.0 : 0.0;
      }
    }
    return probabilities;
  }

  const auto [unused_sign, endpoint_order] = displacement(a, b);
  static_cast<void>(unused_sign);
  const double denominator = scaled_modified_bessel_i(endpoint_order, total_argument);
  if (denominator <= 0.0) {
    throw std::runtime_error("Bessel denominator underflowed");
  }

  for (std::size_t index = 0; index < coordinates.size(); ++index) {
    const auto [unused_left_sign, left_order] = displacement(a, coordinates[index]);
    const auto [unused_right_sign, right_order] = displacement(coordinates[index], b);
    static_cast<void>(unused_left_sign);
    static_cast<void>(unused_right_sign);
    probabilities[index] = scaled_modified_bessel_i(left_order, left_argument) *
                           scaled_modified_bessel_i(right_order, right_argument) / denominator;
  }
  return probabilities;
}

} // namespace qmc
