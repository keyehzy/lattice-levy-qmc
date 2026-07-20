#ifndef QMC_STITCH_MATCHING_HPP
#define QMC_STITCH_MATCHING_HPP

#include "qmc/random.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace qmc::detail {

inline constexpr std::size_t kMaxStitchStrands = 8;
using StitchMatching = std::array<std::size_t, kMaxStitchStrands>;

// Owns a validated row-major log-weight matrix and its permanent recursion.
class PreparedPermanent {
public:
  PreparedPermanent(std::span<const double> log_weights, std::size_t strand_count);

  [[nodiscard]] std::size_t strand_count() const noexcept { return strand_count_; }
  [[nodiscard]] double log_total_weight() const noexcept { return permanent_table_[0]; }

  // Draws sigma with probability proportional to
  // product_i exp(log_weights[i * strand_count + sigma[i]]).
  [[nodiscard]] StitchMatching sample(Random &random) const;

private:
  static constexpr std::size_t kMaxWeightCount = kMaxStitchStrands * kMaxStitchStrands;
  static constexpr std::size_t kMaxStateCount = std::size_t{1} << kMaxStitchStrands;

  std::size_t strand_count_;
  std::array<double, kMaxWeightCount> log_weights_{};
  std::array<double, kMaxStateCount> permanent_table_{};
};

} // namespace qmc::detail

#endif
