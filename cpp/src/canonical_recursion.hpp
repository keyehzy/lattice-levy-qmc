#ifndef QMC_CANONICAL_RECURSION_HPP
#define QMC_CANONICAL_RECURSION_HPP

#include <span>
#include <vector>

namespace qmc::detail {

// Computes log Z_n for n in [0, N] from log z_l for l in [1, N]. Index zero
// of log_cycle_weights is unused. The returned base case is log Z_0 = 0.
[[nodiscard]] std::vector<double>
canonical_log_partitions(std::span<const double> log_cycle_weights);

} // namespace qmc::detail

#endif
