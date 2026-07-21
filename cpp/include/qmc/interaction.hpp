#ifndef QMC_INTERACTION_HPP
#define QMC_INTERACTION_HPP

#include "qmc/continuous_configuration.hpp"
#include "qmc/interacting_model.hpp"

#include <cstddef>

namespace qmc {

// Exact integral int_0^beta d tau sum_x n_x(tau)(n_x(tau)-1)/2.
[[nodiscard]] double pair_overlap_time(const ContinuousConfiguration &configuration);

// Interaction observables derived together from one pair-overlap evaluation.
// The action excludes the fixed-N chemical-potential constant. The
// configuration provenance must match model.free.
struct InteractionMeasurement {
  double action = 0.0;
  double pair_overlap_time = 0.0;
  double double_occupancy_per_site = 0.0;
  double kinetic_energy = 0.0;
  double interaction_energy = 0.0;
  double total_energy = 0.0;
  std::size_t event_count = 0;
};

// Computes the complete bundle with one full pair-overlap sweep.
[[nodiscard]] InteractionMeasurement
measure_interaction(const ContinuousConfiguration &configuration, const InteractingModel &model);

// The chemical-potential term is constant in this fixed-N ensemble.
[[nodiscard]] double interaction_action(const ContinuousConfiguration &configuration,
                                        const InteractingModel &model,
                                        double chemical_potential = 0.0);
[[nodiscard]] double kinetic_energy_estimator(const ContinuousConfiguration &configuration);
[[nodiscard]] double interaction_energy_estimator(const ContinuousConfiguration &configuration,
                                                  const InteractingModel &model);
[[nodiscard]] double total_energy_estimator(const ContinuousConfiguration &configuration,
                                            const InteractingModel &model);
[[nodiscard]] double double_occupancy_per_site(const ContinuousConfiguration &configuration);

} // namespace qmc

#endif
