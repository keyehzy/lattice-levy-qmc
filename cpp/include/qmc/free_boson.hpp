#ifndef QMC_FREE_BOSON_HPP
#define QMC_FREE_BOSON_HPP

#include "qmc/free_numerics.hpp"
#include "qmc/model.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <vector>

namespace qmc {

using Cycle = std::vector<ParticleId>;

struct FreeBosonTable {
  // log_z[ell] = log Z_1(ell*beta), with index zero unused.
  std::vector<double> log_z;
  // Canonical recursion values, with log_Z[0] = 0.
  std::vector<double> log_Z;
};

// Exact finite-momentum log trace for a nonnegative imaginary-time duration.
[[nodiscard]] double log_one_particle_trace(double duration, Coord linear_size,
                                            std::size_t dimension, double hopping);
[[nodiscard]] double log_one_particle_trace(double duration, const Model &model);

// Computes all one-particle cycle weights and canonical partitions through N.
[[nodiscard]] FreeBosonTable canonical_table(const Model &model);

// Samples directed labeled permutation cycles from a matching canonical table.
[[nodiscard]] std::vector<Cycle> sample_cycle_labels(std::size_t particle_count,
                                                     const FreeBosonTable &table, Random &random);

// Samples w with weight I_{|w|*linear_size}(2*hopping*duration).
[[nodiscard]] Coord sample_winding_1d(Coord linear_size, double duration, double hopping,
                                      Random &random,
                                      const NumericalOptions &options = NumericalOptions{});

} // namespace qmc

#endif
