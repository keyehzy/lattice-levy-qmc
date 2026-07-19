#ifndef QMC_FREE_BOSON_HPP
#define QMC_FREE_BOSON_HPP

#include "qmc/free_numerics.hpp"
#include "qmc/model.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qmc {

using Cycle = std::vector<ParticleId>;

// One validated free model and the canonical recursion derived from it. Prefix
// queries through any particle count not exceeding model().particle_count reuse
// the same physical parameters and recursion.
class CanonicalEnsemble {
public:
  explicit CanonicalEnsemble(Model model);

  [[nodiscard]] const Model &model() const noexcept { return model_; }
  // log Z_1(ell*beta), with index zero unused and set to negative infinity.
  [[nodiscard]] std::span<const double> log_cycle_weights() const noexcept { return log_z_; }
  // Canonical log partitions, with log_partitions()[0] == 0.
  [[nodiscard]] std::span<const double> log_partitions() const noexcept { return log_Z_; }
  [[nodiscard]] double log_partition(std::size_t particle_count) const;

  // Samples directed labeled permutation cycles for the ensemble particle count.
  [[nodiscard]] std::vector<Cycle> sample_cycles(Random &random) const;
  // Reuses a canonical prefix. particle_count must not exceed model().particle_count.
  [[nodiscard]] std::vector<Cycle> sample_cycles(std::size_t particle_count, Random &random) const;

private:
  Model model_;
  std::vector<double> log_z_;
  std::vector<double> log_Z_;
};

// Exact finite-momentum log trace for a nonnegative imaginary-time duration.
[[nodiscard]] double log_one_particle_trace(double duration, Coord linear_size,
                                            std::size_t dimension, double hopping);
[[nodiscard]] double log_one_particle_trace(double duration, const Model &model);

// Samples w with weight I_{|w|*linear_size}(2*hopping*duration).
[[nodiscard]] Coord sample_winding_1d(Coord linear_size, double duration, double hopping,
                                      Random &random,
                                      const NumericalOptions &options = NumericalOptions{});

} // namespace qmc

#endif
