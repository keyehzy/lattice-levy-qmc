#ifndef QMC_SRC_ADAPTIVE_DISCRETE_SUPPORT_HPP
#define QMC_SRC_ADAPTIVE_DISCRETE_SUPPORT_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

namespace qmc::detail {

struct SupportGrowth {
  std::size_t minimum;
  double factor;
};

// Shared control state for truncated discrete laws. Distribution-specific
// weights and tail bounds remain at their call sites; this value owns only the
// growth policy, hard work limit, included mass, and relative-tail decision.
class AdaptiveDiscreteSupport {
public:
  AdaptiveDiscreteSupport(const std::size_t initial_support, const std::size_t maximum_support,
                          const double tail_tolerance, const char *limit_error,
                          const char *overflow_error, const char *mass_error)
      : support_(initial_support), maximum_support_(maximum_support),
        log_tolerance_(validated_log_tolerance(tail_tolerance)), limit_error_(limit_error),
        overflow_error_(overflow_error), mass_error_(mass_error) {
    if (support_ > maximum_support_) {
      throw std::runtime_error(limit_error_);
    }
  }

  [[nodiscard]] std::size_t support() const noexcept { return support_; }
  [[nodiscard]] double log_included_mass() const noexcept { return log_included_mass_; }

  void add_weight(const double weight, const double multiplicity = 1.0) {
    if (!std::isfinite(weight) || weight < 0.0 || !std::isfinite(multiplicity) ||
        multiplicity <= 0.0) {
      throw std::runtime_error(mass_error_);
    }
    if (weight == 0.0) {
      return;
    }
    add_log_weight(std::log(weight) + std::log(multiplicity));
  }

  void add_log_weight(const double log_weight) {
    if (std::isnan(log_weight) || log_weight == std::numeric_limits<double>::infinity()) {
      throw std::runtime_error(mass_error_);
    }
    log_included_mass_ = log_add_exp(log_included_mass_, log_weight);
  }

  // A missing bound means the distribution-specific ratio test is not yet
  // contractive. Negative infinity represents an exactly zero omitted tail.
  [[nodiscard]] bool tail_is_controlled(const std::optional<double> log_tail_bound) const {
    if (log_included_mass_ == -std::numeric_limits<double>::infinity()) {
      throw std::runtime_error(mass_error_);
    }
    if (!log_tail_bound.has_value() || std::isnan(*log_tail_bound) ||
        *log_tail_bound == std::numeric_limits<double>::infinity()) {
      return false;
    }
    return *log_tail_bound - log_included_mass_ <= log_tolerance_;
  }

  void grow(const SupportGrowth growth) {
    if (growth.minimum == 0 || !std::isfinite(growth.factor) || growth.factor <= 1.0) {
      throw std::logic_error("adaptive support growth must be positive and expanding");
    }
    if (support_ > std::numeric_limits<std::size_t>::max() - growth.minimum) {
      throw std::overflow_error(overflow_error_);
    }
    const std::size_t additive = support_ + growth.minimum;
    const double scaled = (growth.factor * static_cast<double>(support_)) + 1.0;
    if (!std::isfinite(scaled) ||
        scaled > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error(overflow_error_);
    }
    const std::size_t next = std::max(additive, static_cast<std::size_t>(scaled));
    if (next > maximum_support_) {
      throw std::runtime_error(limit_error_);
    }
    support_ = next;
  }

private:
  [[nodiscard]] static double validated_log_tolerance(const double tail_tolerance) {
    if (!std::isfinite(tail_tolerance) || tail_tolerance <= 0.0 || tail_tolerance >= 1.0) {
      throw std::invalid_argument("tail_tolerance must be finite and lie in (0, 1)");
    }
    return std::log(tail_tolerance);
  }

  [[nodiscard]] static double log_add_exp(const double left, const double right) noexcept {
    if (left == -std::numeric_limits<double>::infinity()) {
      return right;
    }
    if (right == -std::numeric_limits<double>::infinity()) {
      return left;
    }
    const double maximum = std::max(left, right);
    return maximum + std::log1p(std::exp(std::min(left, right) - maximum));
  }

  std::size_t support_;
  const std::size_t maximum_support_;
  const double log_tolerance_;
  double log_included_mass_ = -std::numeric_limits<double>::infinity();
  const char *const limit_error_;
  const char *const overflow_error_;
  const char *const mass_error_;
};

} // namespace qmc::detail

#endif
