#include "stitch_matching.hpp"

#include <algorithm>
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

std::vector<double> log_permanent_table(const std::span<const double> log_weights,
                                        const std::size_t strand_count) {
  validate_log_weights(log_weights, strand_count);
  const std::size_t state_count = std::size_t{1} << strand_count;
  const std::size_t full_mask = state_count - 1;
  std::vector<double> table(state_count, -std::numeric_limits<double>::infinity());
  table[full_mask] = 0.0;

  for (std::size_t mask = full_mask; mask-- > 0;) {
    const auto row = static_cast<std::size_t>(std::popcount(mask));
    double total = -std::numeric_limits<double>::infinity();
    for (std::size_t column = 0; column < strand_count; ++column) {
      const std::size_t column_bit = std::size_t{1} << column;
      if ((mask & column_bit) != 0) {
        continue;
      }
      const double term = log_weights[(row * strand_count) + column] + table[mask | column_bit];
      total = log_add_exp(total, term);
    }
    table[mask] = total;
  }
  return table;
}

std::vector<std::size_t> sample_permanent_matching(const std::span<const double> log_weights,
                                                   const std::size_t strand_count,
                                                   const std::span<const double> permanent_table,
                                                   Random &random) {
  validate_log_weights(log_weights, strand_count);
  const std::size_t state_count = std::size_t{1} << strand_count;
  if (permanent_table.size() != state_count) {
    throw std::invalid_argument("matching weights and permanent table have incompatible sizes");
  }
  for (const double value : permanent_table) {
    if (std::isnan(value) || value == std::numeric_limits<double>::infinity()) {
      throw std::invalid_argument("permanent table may be finite or negative infinity only");
    }
  }
  if (!std::isfinite(permanent_table[0])) {
    throw std::runtime_error("all stitch matchings have zero free weight");
  }

  std::vector<std::size_t> matching(strand_count);
  std::size_t mask = 0;
  for (std::size_t row = 0; row < strand_count; ++row) {
    std::vector<std::size_t> columns;
    std::vector<double> candidate_log_weights;
    columns.reserve(strand_count - row);
    candidate_log_weights.reserve(strand_count - row);
    for (std::size_t column = 0; column < strand_count; ++column) {
      const std::size_t column_bit = std::size_t{1} << column;
      if ((mask & column_bit) != 0) {
        continue;
      }
      columns.push_back(column);
      candidate_log_weights.push_back(log_weights[(row * strand_count) + column] +
                                      permanent_table[mask | column_bit]);
    }
    const std::size_t selected = random.discrete_log_index(candidate_log_weights);
    const std::size_t column = columns[selected];
    matching[row] = column;
    mask |= std::size_t{1} << column;
  }
  return matching;
}

} // namespace qmc::detail
