#ifndef QMC_CONTINUOUS_CONFIGURATION_HPP
#define QMC_CONTINUOUS_CONFIGURATION_HPP

#include "qmc/free_boson.hpp"
#include "qmc/model.hpp"
#include "qmc/path.hpp"
#include "qmc/permutation.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qmc {

namespace detail {
class AcceptedChainState;
}

// Canonical bosonic world-line state with one duration-beta path per label,
// immutable physical provenance, and one authoritative topology. Construction
// validates the complete state; public access is read-only.
class ContinuousConfiguration {
public:
  ContinuousConfiguration(Model model, Permutation topology,
                          std::vector<ContinuousPath> worldlines);

  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] const Permutation &topology() const noexcept { return topology_; }
  // The returned span borrows from this configuration and remains valid until
  // the configuration is assigned a different value or destroyed.
  [[nodiscard]] std::span<const ContinuousPath> worldlines() const noexcept { return worldlines_; }
  [[nodiscard]] const ContinuousPath &path(ParticleId label) const;

  // Expensive diagnostic audit of model provenance, path shape, duration,
  // topology, and endpoint connectivity.
  void validate() const;
  [[nodiscard]] std::size_t event_count() const;
  [[nodiscard]] std::vector<std::size_t> cycle_lengths() const;
  [[nodiscard]] std::vector<Site> positions_at(double tau) const;
  [[nodiscard]] Site total_winding() const;

  [[nodiscard]] bool operator==(const ContinuousConfiguration &) const = default;

private:
  friend class detail::AcceptedChainState;

  void replace_topology(Permutation topology_value) noexcept;

  Model model_;
  Permutation topology_;
  std::vector<ContinuousPath> worldlines_;
};

// Rotates all closed permutation loops by shift in imaginary time. Events at
// the new seam are retained at time zero, preserving right-continuity.
[[nodiscard]] ContinuousConfiguration
rotate_configuration_time_origin(const ContinuousConfiguration &state, double shift);

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
