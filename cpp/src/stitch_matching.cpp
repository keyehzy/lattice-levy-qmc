#include "stitch_matching.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qmc::detail {
namespace {

void validate_log_weights(const std::span<const double> log_weights,
                          const std::size_t strand_count) {
  if (strand_count == 0 || strand_count > kMaxStitchStrands) {
    throw std::invalid_argument("permanent recursion supports between 1 and 8 strands");
  }
  if (log_weights.size() != strand_count * strand_count) {
    throw std::invalid_argument("stitch matching weights must form a square matrix");
  }
  for (const double weight : log_weights) {
    if (std::isnan(weight) || weight == std::numeric_limits<double>::infinity()) {
      throw std::invalid_argument("stitch log weights may be finite or negative infinity only");
    }
  }
}

double log_add_exp(const double left, const double right) {
  if (left == -std::numeric_limits<double>::infinity()) {
    return right;
  }
  if (right == -std::numeric_limits<double>::infinity()) {
    return left;
  }
  const double maximum = std::max(left, right);
  return maximum + std::log1p(std::exp(std::min(left, right) - maximum));
}

} // namespace

PreparedPermanent::PreparedPermanent(const std::span<const double> log_weights,
                                     const std::size_t strand_count)
    : strand_count_(strand_count) {
  validate_log_weights(log_weights, strand_count);
  std::ranges::copy(log_weights, log_weights_.begin());

  const std::size_t state_count = std::size_t{1} << strand_count;
  const std::size_t full_mask = state_count - 1;
  permanent_table_.fill(-std::numeric_limits<double>::infinity());
  permanent_table_[full_mask] = 0.0;

  for (std::size_t mask = full_mask; mask-- > 0;) {
    const auto row = static_cast<std::size_t>(std::popcount(mask));
    double total = -std::numeric_limits<double>::infinity();
    for (std::size_t column = 0; column < strand_count; ++column) {
      const std::size_t column_bit = std::size_t{1} << column;
      if ((mask & column_bit) != 0) {
        continue;
      }
      const double term =
          log_weights_[(row * strand_count) + column] + permanent_table_[mask | column_bit];
      total = log_add_exp(total, term);
    }
    if (std::isnan(total) || total == std::numeric_limits<double>::infinity()) {
      throw std::overflow_error("stitch permanent recursion overflowed");
    }
    permanent_table_[mask] = total;
  }

  if (!std::isfinite(permanent_table_[0])) {
    throw std::runtime_error("all stitch matchings have zero free weight");
  }
}

StitchMatching PreparedPermanent::sample(Random &random) const {
  StitchMatching matching{};
  std::array<std::size_t, kMaxStitchStrands> columns{};
  std::array<double, kMaxStitchStrands> candidate_log_weights{};
  std::size_t mask = 0;
  for (std::size_t row = 0; row < strand_count_; ++row) {
    std::size_t candidate_count = 0;
    for (std::size_t column = 0; column < strand_count_; ++column) {
      const std::size_t column_bit = std::size_t{1} << column;
      if ((mask & column_bit) != 0) {
        continue;
      }
      columns[candidate_count] = column;
      candidate_log_weights[candidate_count] =
          log_weights_[(row * strand_count_) + column] + permanent_table_[mask | column_bit];
      ++candidate_count;
    }
    const std::size_t selected = random.discrete_log_index(
        std::span<const double>(candidate_log_weights.data(), candidate_count));
    const std::size_t column = columns[selected];
    matching[row] = column;
    mask |= std::size_t{1} << column;
  }
  return matching;
}

} // namespace qmc::detail
