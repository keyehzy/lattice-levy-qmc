#ifndef QMC_CONTINUOUS_CONFIGURATION_HPP
#define QMC_CONTINUOUS_CONFIGURATION_HPP

#include "qmc/free_boson.hpp"
#include "qmc/model.hpp"
#include "qmc/path.hpp"
#include "qmc/permutation.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <vector>

namespace qmc {

// Canonical bosonic world-line state with one duration-beta path per label and
// one authoritative, validated topology.
class ContinuousConfiguration {
public:
  ContinuousConfiguration() = default;
  ContinuousConfiguration(Permutation topology_value, std::vector<ContinuousPath> worldline_values,
                          double log_partition);

  std::vector<ContinuousPath> worldlines;
  double log_Z0_N = 0.0;

  [[nodiscard]] const Permutation &topology() const noexcept { return topology_; }
  void validate(const Model &model) const;
  [[nodiscard]] std::size_t event_count() const;
  [[nodiscard]] std::vector<std::size_t> cycle_lengths() const;
  [[nodiscard]] std::vector<Site> positions_at(double tau, const Model &model) const;
  [[nodiscard]] Site total_winding(const Model &model) const;

private:
  friend class InteractingSampler;

  void replace_topology(Permutation topology_value) noexcept;

  Permutation topology_;
};

// Rotates all closed permutation loops by shift in imaginary time. Events at
// the new seam are retained at time zero, preserving right-continuity.
[[nodiscard]] ContinuousConfiguration
rotate_configuration_time_origin(const ContinuousConfiguration &state, const Model &model,
                                 double shift);

// Samples the continuous event representation of a retained canonical ensemble.
[[nodiscard]] ContinuousConfiguration
sample_ideal_continuous_configuration(const CanonicalEnsemble &ensemble, Random &random,
                                      const NumericalOptions &options = NumericalOptions{});

// One-off convenience overload; repeated workflows should retain a CanonicalEnsemble.
[[nodiscard]] ContinuousConfiguration
sample_ideal_continuous_configuration(const Model &model, Random &random,
                                      const NumericalOptions &options = NumericalOptions{});

} // namespace qmc

#endif
