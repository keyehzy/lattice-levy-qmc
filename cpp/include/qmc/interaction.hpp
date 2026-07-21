#ifndef QMC_INTERACTION_HPP
#define QMC_INTERACTION_HPP

#include "qmc/continuous_configuration.hpp"
#include "qmc/interacting_model.hpp"

namespace qmc {

// Exact integral int_0^beta d tau sum_x n_x(tau)(n_x(tau)-1)/2.
[[nodiscard]] double pair_overlap_time(const ContinuousConfiguration &configuration);

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
