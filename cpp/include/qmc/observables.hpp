#ifndef QMC_OBSERVABLES_HPP
#define QMC_OBSERVABLES_HPP

#include "qmc/configuration.hpp"
#include "qmc/free_boson.hpp"
#include "qmc/model.hpp"
#include "qmc/torus_layout.hpp"

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

[[nodiscard]] CanonicalThermodynamics canonical_thermodynamics(const CanonicalEnsemble &ensemble);
[[nodiscard]] CanonicalThermodynamics canonical_thermodynamics(const Model &model);

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
[[nodiscard]] MomentumDistribution momentum_distribution(const CanonicalEnsemble &ensemble);
[[nodiscard]] MomentumDistribution momentum_distribution(const Model &model);

struct OneBodyDensityPoint {
  // Torus displacement components in [0, L).
  Site displacement;
  double value = 0.0;
};

// Translation-invariant <a_r^dagger a_{r+delta}>, returned for every delta.
[[nodiscard]] std::vector<OneBodyDensityPoint>
one_body_density_matrix(const CanonicalEnsemble &ensemble);
[[nodiscard]] std::vector<OneBodyDensityPoint> one_body_density_matrix(const Model &model);

struct ExactCycleStatistics {
  // All arrays are indexed by cycle length; index zero is unused and set to zero.
  std::vector<double> expected_cycle_count;
  std::vector<double> expected_particles;
  std::vector<double> particle_probability;
};

[[nodiscard]] ExactCycleStatistics exact_cycle_statistics(const CanonicalEnsemble &ensemble);
// A prefix query uses the ensemble's beta, lattice, and hopping and rejects counts
// greater than ensemble.model().particle_count().
[[nodiscard]] ExactCycleStatistics exact_cycle_statistics(const CanonicalEnsemble &ensemble,
                                                          std::size_t particle_count);
[[nodiscard]] ExactCycleStatistics exact_cycle_statistics(const Model &model);
[[nodiscard]] std::vector<std::size_t>
sampled_cycle_histogram(const IdealBosonConfiguration &configuration);
[[nodiscard]] std::size_t longest_cycle_length(const IdealBosonConfiguration &configuration);

// Sum of the per-cycle winding vectors derived from authoritative covering paths.
[[nodiscard]] Site total_winding(const IdealBosonConfiguration &configuration);

// Exact log Z_N for a boundary twist phi. Each component enters as q_alpha + phi_alpha/L.
[[nodiscard]] double log_canonical_partition_twisted(const CanonicalEnsemble &ensemble,
                                                     std::span<const double> twist);
[[nodiscard]] double log_canonical_partition_twisted(const Model &model,
                                                     std::span<const double> twist);

// Exact zero-twist curvature d^2 F_N/d phi_axis^2. It equals <W_axis^2>/beta.
[[nodiscard]] double twist_free_energy_curvature(const CanonicalEnsemble &ensemble,
                                                 std::size_t axis);
[[nodiscard]] double twist_free_energy_curvature(const Model &model, std::size_t axis);

// Immutable provenance for an exact retained-time grid on one torus. beta may
// be zero, but transforms that divide by beta reject that boundary value.
class RetainedGrid {
public:
  // Throws invalid_argument unless beta is finite/nonnegative and time_points is positive.
  RetainedGrid(double beta, TorusLayout layout, std::size_t time_points);

  [[nodiscard]] double beta() const noexcept { return beta_; }
  [[nodiscard]] const TorusLayout &layout() const noexcept { return layout_; }
  [[nodiscard]] std::size_t time_points() const noexcept { return time_points_; }

  bool operator==(const RetainedGrid &) const = default;

private:
  double beta_;
  TorusLayout layout_;
  std::size_t time_points_;
};

// Reusable, owning measurement state derived from one retained ideal
// configuration. Positions are reduced to physical SiteId values once and the
// duplicate beta endpoint is excluded.
class RetainedMeasurementContext {
public:
  explicit RetainedMeasurementContext(const IdealBosonConfiguration &configuration);

  [[nodiscard]] const RetainedGrid &grid() const noexcept { return grid_; }
  [[nodiscard]] std::size_t particle_count() const noexcept { return particle_count_; }
  // Returns one particle-ordered retained slice and throws out_of_range for an
  // index outside [0, grid().time_points()).
  [[nodiscard]] std::span<const SiteId> positions_at(std::size_t time_index) const;

  bool operator==(const RetainedMeasurementContext &) const = default;

private:
  RetainedGrid grid_;
  std::size_t particle_count_;
  std::vector<SiteId> positions_;
};

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
equal_time_observables(const RetainedMeasurementContext &context);
// One-off convenience overload; construct a RetainedMeasurementContext when
// evaluating more than one retained observable for the same configuration.
[[nodiscard]] EqualTimeObservables
equal_time_observables(const IdealBosonConfiguration &configuration);

// Averages equal-time estimators over compatible retained configurations.
// Each observed context must have the construction grid and particle count.
class EqualTimeAccumulator {
public:
  EqualTimeAccumulator(RetainedGrid grid, std::size_t particle_count);

  [[nodiscard]] const RetainedGrid &grid() const noexcept { return grid_; }
  [[nodiscard]] std::size_t particle_count() const noexcept { return particle_count_; }
  [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }

  void observe(const RetainedMeasurementContext &context);
  // Throws logic_error when no sample has been observed.
  [[nodiscard]] EqualTimeObservables finish() const;

private:
  RetainedGrid grid_;
  std::size_t particle_count_;
  std::size_t sample_count_ = 0;
  EqualTimeObservables sums_;
};

// Shape-safe connected C_nn(delta, tau_j). The estimator averages all retained
// time origins and uses periodic imaginary-time differences. Flat storage uses
// time-major ordering followed by the grid layout's flat displacement.
class ImaginaryTimeDensityCorrelations {
public:
  // Requires exactly grid.time_points()*grid.layout().volume() values. Throws
  // overflow_error when that extent is not representable and invalid_argument
  // when storage has the wrong size.
  ImaginaryTimeDensityCorrelations(RetainedGrid grid, std::vector<double> connected_density);

  [[nodiscard]] const RetainedGrid &grid() const noexcept { return grid_; }
  [[nodiscard]] std::size_t time_points() const noexcept { return grid_.time_points(); }
  [[nodiscard]] std::size_t spatial_points() const noexcept { return grid_.layout().volume(); }
  [[nodiscard]] std::span<const double> connected_density() const noexcept {
    return connected_density_;
  }
  [[nodiscard]] double at(std::size_t time_index, SiteId displacement) const;

  bool operator==(const ImaginaryTimeDensityCorrelations &) const = default;

private:
  RetainedGrid grid_;
  std::vector<double> connected_density_;
};

[[nodiscard]] ImaginaryTimeDensityCorrelations
retained_density_correlations(const RetainedMeasurementContext &context);
// One-off convenience overload; construct a RetainedMeasurementContext when
// evaluating more than one retained observable for the same configuration.
[[nodiscard]] ImaginaryTimeDensityCorrelations
retained_density_correlations(const IdealBosonConfiguration &configuration);

// Averages connected density-correlation estimators over compatible retained
// configurations. Each observed context must have the construction grid and
// particle count.
class RetainedDensityCorrelationAccumulator {
public:
  RetainedDensityCorrelationAccumulator(RetainedGrid grid, std::size_t particle_count);

  [[nodiscard]] const RetainedGrid &grid() const noexcept { return grid_; }
  [[nodiscard]] std::size_t particle_count() const noexcept { return particle_count_; }
  [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }

  void observe(const RetainedMeasurementContext &context);
  // Throws logic_error when no sample has been observed.
  [[nodiscard]] ImaginaryTimeDensityCorrelations finish() const;

private:
  RetainedGrid grid_;
  std::size_t particle_count_;
  std::size_t sample_count_ = 0;
  std::vector<double> connected_density_sums_;
};

struct MatsubaraDensityCorrelations {
  std::vector<double> frequencies;
  std::size_t momentum_points = 0;
  // Grid approximation (beta/M) sum_j sum_delta exp(i*omega_n*tau_j-i*q*delta) C(delta,tau_j),
  // indexed by n*momentum_points + flat(q).
  std::vector<std::complex<double>> values;
};

// Uses the input's retained-grid beta and torus layout. Requires beta > 0.
[[nodiscard]] MatsubaraDensityCorrelations
retained_grid_matsubara_transform(const ImaginaryTimeDensityCorrelations &correlations);

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

// Averages retained covering/displacement geometry over compatible ideal
// configurations. Each observed configuration must have the construction grid
// and particle count.
class RetainedGeometryAccumulator {
public:
  RetainedGeometryAccumulator(RetainedGrid grid, std::size_t particle_count);

  [[nodiscard]] const RetainedGrid &grid() const noexcept { return grid_; }
  [[nodiscard]] std::size_t particle_count() const noexcept { return particle_count_; }
  [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }

  void observe(const IdealBosonConfiguration &configuration);
  // Throws logic_error when no sample has been observed.
  [[nodiscard]] RetainedGeometryObservables finish() const;

private:
  RetainedGrid grid_;
  std::size_t particle_count_;
  std::size_t sample_count_ = 0;
  RetainedGeometryObservables sums_;
};

struct RetainedCycleGeometry {
  std::size_t length = 0;
  Site winding;
  // Covering-space radius of gyration and maximum squared radius over retained cycle points.
  double radius_of_gyration_squared = 0.0;
  double maximum_radius_squared = 0.0;
};

[[nodiscard]] std::vector<RetainedCycleGeometry>
retained_cycle_geometry(const IdealBosonConfiguration &configuration);

struct SampledCycleStatistics {
  // All arrays are indexed by cycle length. Index zero is unused except that
  // longest_cycle_probability[0] is one for an empty system.
  std::vector<double> mean_cycle_count;
  std::vector<double> mean_particles;
  // Geometry means are conditioned on observing a cycle of the indexed length;
  // a length that was never observed retains zero.
  std::vector<double> mean_cycle_winding_squared;
  std::vector<double> mean_radius_of_gyration_squared;
  std::vector<double> mean_maximum_radius_squared;
  std::vector<double> longest_cycle_probability;
  // Mean particle fraction in cycles of length at least ceil(N/2).
  double macroscopic_cycle_fraction = 0.0;
};

// Averages cycle counts and retained cycle geometry over compatible ideal
// configurations. Each observed configuration must have the construction grid
// and particle count.
class CycleStatisticsAccumulator {
public:
  CycleStatisticsAccumulator(RetainedGrid grid, std::size_t particle_count);

  [[nodiscard]] const RetainedGrid &grid() const noexcept { return grid_; }
  [[nodiscard]] std::size_t particle_count() const noexcept { return particle_count_; }
  [[nodiscard]] std::size_t macroscopic_cycle_threshold() const noexcept {
    return macroscopic_cycle_threshold_;
  }
  [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }

  // Returns the per-sample cycle geometry so callers can retain traces without
  // evaluating the covering paths a second time.
  std::vector<RetainedCycleGeometry> observe(const IdealBosonConfiguration &configuration);
  // Throws logic_error when no sample has been observed.
  [[nodiscard]] SampledCycleStatistics finish() const;

private:
  RetainedGrid grid_;
  std::size_t particle_count_;
  std::size_t macroscopic_cycle_threshold_;
  std::size_t sample_count_ = 0;
  SampledCycleStatistics sums_;
  std::vector<double> cycle_occurrences_;
};

struct WindingStatistics {
  std::vector<double> second_moment;
  std::vector<double> fourth_moment;
  double nonzero_probability = 0.0;
};

// Averages total covering-space winding over compatible ideal configurations.
// Each observed configuration must have the construction grid and particle
// count.
class WindingAccumulator {
public:
  WindingAccumulator(RetainedGrid grid, std::size_t particle_count);

  [[nodiscard]] const RetainedGrid &grid() const noexcept { return grid_; }
  [[nodiscard]] std::size_t particle_count() const noexcept { return particle_count_; }
  [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }

  // Returns the per-sample total winding so callers can retain traces without
  // deriving it a second time.
  Site observe(const IdealBosonConfiguration &configuration);
  // Throws logic_error when no sample has been observed.
  [[nodiscard]] WindingStatistics finish() const;

private:
  RetainedGrid grid_;
  std::size_t particle_count_;
  std::size_t sample_count_ = 0;
  WindingStatistics sums_;
};

} // namespace qmc

#endif
