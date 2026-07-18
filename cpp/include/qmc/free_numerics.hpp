#ifndef QMC_FREE_NUMERICS_HPP
#define QMC_FREE_NUMERICS_HPP

#include "qmc/model.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qmc {

using Site = std::vector<Coord>;
using CoveringPath = std::vector<Site>;

// Euclidean reduction into [0, linear_size), including for negative coordinates.
[[nodiscard]] Coord torus_mod(Coord coordinate, Coord linear_size);
[[nodiscard]] double log_sum_exp(std::span<const double> values);
// Returns exp(-argument) I_order(argument) through GSL's scaled implementation.
[[nodiscard]] double scaled_modified_bessel_i(std::uint64_t order, double argument);

// Samples the smaller conditioned Poisson count for endpoint displacement abs_delta.
[[nodiscard]] std::uint64_t
sample_bessel_pair_count(std::uint64_t abs_delta, double lambda, Random &random,
                         const NumericalOptions &options = NumericalOptions{});

// Samples a covering-space midpoint between a and b over nonnegative left/right durations.
[[nodiscard]] Coord
sample_midpoint_covering_1d(Coord a, Coord b, double tau_left, double tau_right, double hopping,
                            Random &random, const NumericalOptions &options = NumericalOptions{});

// Coordinate-factorized multidimensional version of sample_midpoint_covering_1d.
[[nodiscard]] Site sample_midpoint_covering(const Site &a, const Site &b, double tau_left,
                                            double tau_right, double hopping, Random &random,
                                            const NumericalOptions &options = NumericalOptions{});

// Samples steps + 1 exact bridge observations, including both covering-space endpoints.
[[nodiscard]] CoveringPath
sample_bridge_covering(const Site &a, const Site &b, double total_time, std::size_t steps,
                       double hopping, Random &random,
                       const NumericalOptions &options = NumericalOptions{});

// Returns exp(-2*hopping*duration) times the finite-ring kernel.
[[nodiscard]] double periodic_kernel_scaled_1d(Coord delta, double duration, Coord linear_size,
                                               double hopping);

// Samples a physical midpoint in [0, linear_size), implicitly summing all winding sectors.
[[nodiscard]] Coord sample_midpoint_torus_1d(Coord a, Coord b, double tau_left, double tau_right,
                                             Coord linear_size, double hopping, Random &random);

// Evaluates the exactly normalized infinite-lattice midpoint PMF at supplied coordinates.
[[nodiscard]] std::vector<double> exact_midpoint_pmf_window(Coord a, Coord b, double tau_left,
                                                            double tau_right, double hopping,
                                                            std::span<const Coord> coordinates);

} // namespace qmc

#endif
