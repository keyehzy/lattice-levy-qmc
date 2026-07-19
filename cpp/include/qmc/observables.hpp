#ifndef QMC_OBSERVABLES_HPP
#define QMC_OBSERVABLES_HPP

#include "qmc/configuration.hpp"
#include "qmc/free_boson.hpp"
#include "qmc/model.hpp"

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace qmc {

// Exact fixed-N thermodynamics, indexed by particle number from zero through model.N.
// The addition chemical potential at N=0 is NaN. This table requires beta > 0 and k_B = 1.
struct CanonicalThermodynamics {
  std::vector<double> free_energy;
  std::vector<double> energy;
  std::vector<double> heat_capacity;
  std::vector<double> entropy;
  std::vector<double> addition_chemical_potential;
};

[[nodiscard]] CanonicalThermodynamics canonical_thermodynamics(const Model &model,
                                                               const FreeBosonTable &table);

struct MomentumMode {
  // Integer components k_alpha in q_alpha = 2*pi*k_alpha/L.
  std::vector<std::size_t> indices;
  std::vector<double> wavevector;
  double energy = 0.0;
  double occupation = 0.0;
  double occupation_variance = 0.0;
};

struct MomentumDistribution {
  std::vector<MomentumMode> modes;
  double condensate_occupation = 0.0;
  double condensate_fraction = 0.0;
  double condensate_density = 0.0;
  // Standard finite-lattice second-moment length from n(0)/n(q_min), averaged over axes.
  double coherence_length = 0.0;
  double kinetic_energy = 0.0;
};

// Returns every finite-torus momentum mode. Mode zero is q=(0,...,0).
[[nodiscard]] MomentumDistribution momentum_distribution(const Model &model,
                                                         const FreeBosonTable &table);

struct OneBodyDensityPoint {
  // Torus displacement components in [0, L).
  Site displacement;
  double value = 0.0;
};

// Translation-invariant <a_r^dagger a_{r+delta}>, returned for every delta.
[[nodiscard]] std::vector<OneBodyDensityPoint> one_body_density_matrix(const Model &model,
                                                                       const FreeBosonTable &table);

struct ExactCycleStatistics {
  // All arrays are indexed by cycle length; index zero is unused and set to zero.
  std::vector<double> expected_cycle_count;
  std::vector<double> expected_particles;
  std::vector<double> particle_probability;
};

[[nodiscard]] ExactCycleStatistics exact_cycle_statistics(std::size_t particle_count,
                                                          const FreeBosonTable &table);
[[nodiscard]] std::vector<std::size_t>
sampled_cycle_histogram(const IdealBosonConfiguration &configuration);
[[nodiscard]] std::size_t longest_cycle_length(const IdealBosonConfiguration &configuration);

// Sum of the explicitly retained per-cycle covering-space winding vectors.
[[nodiscard]] Site total_winding(const IdealBosonConfiguration &configuration);

// Exact log Z_N for a boundary twist phi. Each component enters as q_alpha + phi_alpha/L.
[[nodiscard]] double log_canonical_partition_twisted(const Model &model,
                                                     std::span<const double> twist);

// Exact zero-twist curvature d^2 F_N/d phi_axis^2. It equals <W_axis^2>/beta.
[[nodiscard]] double twist_free_energy_curvature(const Model &model, const FreeBosonTable &table,
                                                 std::size_t axis);

struct EqualTimeObservables {
  // Flat site/momentum ordering uses axis zero as the least-significant base-L digit.
  // Values are averaged over the M distinct retained slices; the duplicate beta endpoint
  // is excluded.
  std::vector<double> site_density;
  std::vector<double> pair_correlation;
  std::vector<double> static_structure_factor;
  std::vector<double> onsite_occupation_probability;
  double mean_occupation_squared = 0.0;
  double mean_factorial_occupation = 0.0;
};

[[nodiscard]] EqualTimeObservables
equal_time_observables(const IdealBosonConfiguration &configuration);

struct ImaginaryTimeDensityCorrelations {
  std::size_t time_points = 0;
  std::size_t spatial_points = 0;
  // Connected C_nn(delta, tau_j), indexed by j*spatial_points + flat(delta). The estimator
  // averages all retained time origins and uses periodic time differences on the M-point grid.
  std::vector<double> connected_density;
};

[[nodiscard]] ImaginaryTimeDensityCorrelations
retained_density_correlations(const IdealBosonConfiguration &configuration);

struct MatsubaraDensityCorrelations {
  std::vector<double> frequencies;
  std::size_t momentum_points = 0;
  // Grid approximation (beta/M) sum_j sum_delta exp(i*omega_n*tau_j-i*q*delta) C(delta,tau_j),
  // indexed by n*momentum_points + flat(q).
  std::vector<std::complex<double>> values;
};

[[nodiscard]] MatsubaraDensityCorrelations
retained_grid_matsubara_transform(const Model &model,
                                  const ImaginaryTimeDensityCorrelations &correlations);

struct RetainedGeometryObservables {
  std::size_t time_points = 0;
  std::size_t displacement_points = 0;
  // Same-label covering-space displacement relative to tau=0, averaged over particles.
  std::vector<double> mean_square_displacement;
  // Probability of returning to the same torus site.
  std::vector<double> return_probability;
  // Torus displacement PMF, indexed by j*displacement_points + flat(displacement).
  std::vector<double> displacement_probability;
};

[[nodiscard]] RetainedGeometryObservables
retained_geometry_observables(const IdealBosonConfiguration &configuration);

struct RetainedCycleGeometry {
  std::size_t length = 0;
  Site winding;
  // Covering-space radius of gyration and maximum squared radius over retained cycle points.
  double radius_of_gyration_squared = 0.0;
  double maximum_radius_squared = 0.0;
};

[[nodiscard]] std::vector<RetainedCycleGeometry>
retained_cycle_geometry(const IdealBosonConfiguration &configuration);

} // namespace qmc

#endif
