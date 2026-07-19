#ifndef QMC_STITCH_MATCHING_HPP
#define QMC_STITCH_MATCHING_HPP

#include "qmc/random.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qmc::detail {

inline constexpr std::size_t kMaxStitchStrands = 8;

// Returns F(mask), the log total weight of assigning rows popcount(mask), ...,
// k - 1 to the columns absent from mask. The matrix is row-major.
[[nodiscard]] std::vector<double> log_permanent_table(std::span<const double> log_weights,
                                                      std::size_t strand_count);

// Draws sigma with probability proportional to
// product_i exp(log_weights[i * strand_count + sigma[i]]).
[[nodiscard]] std::vector<std::size_t>
sample_permanent_matching(std::span<const double> log_weights, std::size_t strand_count,
                          std::span<const double> permanent_table, Random &random);

} // namespace qmc::detail

#endif
