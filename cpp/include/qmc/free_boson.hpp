#ifndef QMC_FREE_BOSON_HPP
#define QMC_FREE_BOSON_HPP

#include "qmc/free_numerics.hpp"
#include "qmc/model.hpp"
#include "qmc/permutation.hpp"
#include "qmc/random.hpp"
#include "qmc/torus_layout.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qmc {

// Immutable one-dimensional momentum data and tight-binding dispersion for a
// validated L^d torus. The tables contain q=2*pi*k/L and its sine/cosine for
// k in [0,L); full L^d momentum records remain demand-driven.
class OneParticleSpectrum {
public:
  // Throws invalid_argument for invalid geometry/hopping and overflow_error
  // when L^d is not representable.
  OneParticleSpectrum(Coord linear_size, std::size_t dimension, double hopping);
  explicit OneParticleSpectrum(const Model &model);

  [[nodiscard]] const TorusLayout &layout() const noexcept { return layout_; }
  [[nodiscard]] double hopping() const noexcept { return hopping_; }
  [[nodiscard]] std::span<const double> wavevectors() const noexcept { return wavevectors_; }
  [[nodiscard]] std::span<const double> cosines() const noexcept { return cosines_; }
  [[nodiscard]] std::span<const double> sines() const noexcept { return sines_; }

  // Returns -2*t*sum_axis cos(q_axis). The component count must equal the
  // lattice dimension and every component must lie in [0,L).
  [[nodiscard]] double energy(std::span<const std::size_t> momentum_components) const;

  bool operator==(const OneParticleSpectrum &) const = default;

private:
  TorusLayout layout_;
  double hopping_;
  std::vector<double> wavevectors_;
  std::vector<double> cosines_;
  std::vector<double> sines_;
};

// One validated free model, numerical path policy, and canonical recursion
// derived from them. Prefix queries through any particle count not exceeding
// model().particle_count() reuse the same physical parameters and recursion.
class CanonicalEnsemble {
public:
  explicit CanonicalEnsemble(Model model, NumericalOptions numerical = NumericalOptions{});

  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] const OneParticleSpectrum &spectrum() const noexcept { return spectrum_; }
  [[nodiscard]] const FreePathKernels &free_path_kernels() const noexcept {
    return free_path_kernels_;
  }
  // log Z_1(ell*beta), with index zero unused and set to negative infinity.
  [[nodiscard]] std::span<const double> log_cycle_weights() const noexcept { return log_z_; }
  // Canonical log partitions, with log_partitions()[0] == 0.
  [[nodiscard]] std::span<const double> log_partitions() const noexcept { return log_Z_; }
  [[nodiscard]] double log_partition(std::size_t particle_count) const;

  // Samples directed labeled permutation cycles for the ensemble particle count.
  [[nodiscard]] std::vector<Cycle> sample_cycles(Random &random) const;
  // Reuses a canonical prefix. particle_count must not exceed model().particle_count().
  [[nodiscard]] std::vector<Cycle> sample_cycles(std::size_t particle_count, Random &random) const;

private:
  Model model_;
  FreePathKernels free_path_kernels_;
  OneParticleSpectrum spectrum_;
  std::vector<double> log_z_;
  std::vector<double> log_Z_;
};

// Exact finite-momentum log trace for a nonnegative imaginary-time duration.
[[nodiscard]] double log_one_particle_trace(double duration, const OneParticleSpectrum &spectrum);
[[nodiscard]] double log_one_particle_trace(double duration, Coord linear_size,
                                            std::size_t dimension, double hopping);
[[nodiscard]] double log_one_particle_trace(double duration, const Model &model);

// Samples w with weight I_{|w|*linear_size}(2*hopping*duration).
[[nodiscard]] Coord sample_winding_1d(Coord linear_size, double duration, double hopping,
                                      Random &random,
                                      const NumericalOptions &options = NumericalOptions{});

} // namespace qmc

#endif
