#ifndef QMC_SRC_TORUS_BRIDGE_DISTRIBUTION_HPP
#define QMC_SRC_TORUS_BRIDGE_DISTRIBUTION_HPP

#include "qmc/free_numerics.hpp"

#include <cstddef>
#include <vector>

namespace qmc::detail {

// Prepared exact winding-sector law for one physical torus displacement and
// duration. Construction performs all adaptive-support work; the resulting
// value supplies both the matching normalization and covering-endpoint draws.
class TorusBridgeDistribution {
public:
  TorusBridgeDistribution(SiteId displacement, double duration, const FreePathKernels &kernels);

  [[nodiscard]] double log_normalization() const noexcept { return log_normalization_; }
  [[nodiscard]] Site sample_covering_endpoint(const Site &start, Random &random) const;

private:
  struct AxisDistribution {
    Coord displacement;
    std::vector<Coord> windings;
    std::vector<double> weights;
  };

  Coord linear_size_;
  std::size_t dimension_;
  double log_normalization_ = 0.0;
  std::vector<AxisDistribution> axes_;
};

} // namespace qmc::detail

#endif
