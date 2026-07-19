#ifndef QMC_CONTINUOUS_CONFIGURATION_HPP
#define QMC_CONTINUOUS_CONFIGURATION_HPP

#include "qmc/free_boson.hpp"
#include "qmc/model.hpp"
#include "qmc/path.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qmc {

// Canonical bosonic world-line state with one duration-beta path per label.
struct ContinuousConfiguration {
  std::vector<Cycle> cycles;
  std::vector<ParticleId> permutation;
  std::vector<ContinuousPath> worldlines;
  double log_Z0_N = 0.0;

  void validate(const Model &model) const;
  [[nodiscard]] std::size_t event_count() const;
  [[nodiscard]] std::vector<std::size_t> cycle_lengths() const;
  [[nodiscard]] std::vector<Site> positions_at(double tau, const Model &model) const;
  [[nodiscard]] Site total_winding(const Model &model) const;
};

// Deterministic cycle decomposition, rooted in increasing label order.
[[nodiscard]] std::vector<Cycle> cycles_from_permutation(std::span<const ParticleId> permutation);

// Rotates all closed permutation loops by shift in imaginary time. Events at
// the new seam are retained at time zero, preserving right-continuity.
[[nodiscard]] ContinuousConfiguration
rotate_configuration_time_origin(const ContinuousConfiguration &state, const Model &model,
                                 double shift);

// Samples the continuous event representation of the exact canonical ideal measure.
[[nodiscard]] ContinuousConfiguration
sample_ideal_continuous_configuration(const Model &model, Random &random,
                                      const NumericalOptions &options = NumericalOptions{});

// Reuses a canonical table computed for model. This overload is intended for
// chains that make repeated global ideal proposals.
[[nodiscard]] ContinuousConfiguration
sample_ideal_continuous_configuration(const Model &model, const FreeBosonTable &table,
                                      Random &random,
                                      const NumericalOptions &options = NumericalOptions{});

} // namespace qmc

#endif
