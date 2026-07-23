#ifndef QMC_CONTINUOUS_OBSERVABLES_HPP
#define QMC_CONTINUOUS_OBSERVABLES_HPP

#include "qmc/continuous_configuration.hpp"
#include "qmc/matsubara_modes.hpp"
#include "qmc/torus_layout.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qmc {

class ContinuousMeasurementContext;
class ContinuousParticleModes;
class DensityMatsubaraAccumulator;
class HoppingResponseAccumulator;

// Reusable continuous-time phase plan. The stricter signed-frequency bound is
// a binary64 phase-reduction contract and is intentionally not imposed by the
// geometry-only MatsubaraModeSet.
class ContinuousMatsubaraPlan {
public:
  static constexpr std::int64_t kMaximumAbsoluteFrequencyIndex = 1'048'576;

  // Throws invalid_argument when any |n| exceeds the supported bound and
  // overflow_error/length_error when phase storage is not representable.
  explicit ContinuousMatsubaraPlan(MatsubaraModeSet modes);

  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept { return modes_; }

private:
  friend ContinuousParticleModes
  continuous_particle_modes(const ContinuousMeasurementContext &context,
                            const ContinuousMatsubaraPlan &plan);

  MatsubaraModeSet modes_;
  // Momentum-major, then axis. These immutable factors support physical-site
  // updates and both orientations of bond-midpoint impulses.
  std::vector<std::complex<double>> site_step_phases_;
  std::vector<std::complex<double>> positive_midpoint_phases_;
  std::vector<std::complex<double>> negative_midpoint_phases_;
};

// One physical nearest-neighbor hop in a continuous configuration. Time is in
// [0, beta], sites are reduced onto the torus, and direction retains the
// covering-space sign even when periodic reduction makes both signs coincide.
struct ContinuousHop {
  double time;
  ParticleId particle;
  SiteId departure;
  SiteId arrival;
  Axis axis;
  std::int8_t direction;

  [[nodiscard]] bool operator==(const ContinuousHop &) const = default;
};

// Owning event geometry derived from one valid continuous configuration.
// seam_positions() is the state immediately before events at time zero. Hops
// are globally nondecreasing in time; ties retain particle/path event order and
// are exposed as atomic groups.
class ContinuousMeasurementContext {
public:
  explicit ContinuousMeasurementContext(const ContinuousConfiguration &configuration);

  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] const TorusLayout &layout() const noexcept { return layout_; }
  [[nodiscard]] std::span<const SiteId> seam_positions() const noexcept { return seam_positions_; }
  [[nodiscard]] std::span<const ContinuousHop> hops() const noexcept { return hops_; }
  [[nodiscard]] std::size_t event_group_count() const noexcept {
    return event_group_offsets_.size() - 1;
  }
  // Both group accessors throw std::out_of_range for an unknown group.
  [[nodiscard]] double event_time(std::size_t group) const;
  [[nodiscard]] std::span<const ContinuousHop> hops_at(std::size_t group) const;

private:
  Model model_;
  TorusLayout layout_;
  std::vector<SiteId> seam_positions_;
  std::vector<ContinuousHop> hops_;
  // Includes the terminal offset, so an empty context owns the single value 0.
  std::vector<std::size_t> event_group_offsets_{0};
};

// Unnormalised density-residence and signed hopping-flux amplitudes for one
// continuous configuration. Values use frequency-major Matsubara mode order;
// flux adds one physical-axis component after the mode index. The result owns
// its complete free-model and Fourier provenance. Flux uses oriented bond
// midpoints; canonicalizing -q into [0,L) adds a reciprocal-lattice minus sign
// for a flux component whose momentum along its bond axis is nonzero.
class ContinuousParticleModes {
public:
  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept { return density_.modes(); }
  // Integral d-tau exp(+i*omega*tau) sum_a exp(-i*q*x_a(tau)); units 1/energy.
  // Checked mode accessors throw out_of_range.
  [[nodiscard]] std::complex<double> density(std::size_t frequency, std::size_t momentum) const;
  // Sum over axis hops of s*exp(+i*omega*tau-i*q*(x+s*e_axis/2)); dimensionless.
  [[nodiscard]] std::complex<double> flux(std::size_t frequency, std::size_t momentum,
                                          std::size_t axis) const;
  // Total unsigned hopping-event count on one physical axis.
  [[nodiscard]] std::size_t axis_event_count(std::size_t axis) const;

private:
  friend ContinuousParticleModes
  continuous_particle_modes(const ContinuousMeasurementContext &context,
                            const ContinuousMatsubaraPlan &plan);

  ContinuousParticleModes(Model model, MatsubaraModeSet modes,
                          std::vector<std::complex<double>> density_values,
                          std::vector<std::complex<double>> flux_values,
                          std::vector<std::size_t> axis_event_counts);

  Model model_;
  MatsubaraModeField<std::complex<double>> density_;
  std::vector<std::complex<double>> flux_values_;
  std::vector<std::size_t> axis_event_counts_;
};

// Connected density susceptibility for one homogeneous fixed-particle-number
// ensemble. Values are <|delta rho(q,n)|^2>/(beta*V), have units of inverse
// energy per site, and use the result's frequency-major Matsubara mode order.
// The sampled uncentred mean amplitude is retained as a symmetry diagnostic.
class ContinuousMatsubaraDensityCorrelations {
public:
  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept { return correlations_.modes(); }
  [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }
  // Both checked mode accessors throw out_of_range.
  [[nodiscard]] std::complex<double> mean_amplitude(std::size_t frequency,
                                                    std::size_t momentum) const;
  [[nodiscard]] double at(std::size_t frequency, std::size_t momentum) const {
    return correlations_.at(frequency, momentum);
  }

private:
  friend class DensityMatsubaraAccumulator;

  ContinuousMatsubaraDensityCorrelations(Model model, MatsubaraModeField<double> correlations,
                                         std::vector<std::complex<double>> mean_amplitudes,
                                         std::size_t sample_count);

  Model model_;
  MatsubaraModeField<double> correlations_;
  std::vector<std::complex<double>> mean_amplitudes_;
  std::size_t sample_count_;
};

// Averages exact ContinuousParticleModes density amplitudes for one complete
// free Model and one selected Matsubara mode set. Connectedness uses the exact
// homogeneous fixed-N mean beta*N at (q,n)=(0,0), and zero elsewhere. observe()
// validates all provenance and candidate sums before mutating the accumulator.
class DensityMatsubaraAccumulator {
public:
  // Throws invalid_argument when model beta/layout and modes differ.
  DensityMatsubaraAccumulator(Model model, MatsubaraModeSet modes);

  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept { return modes_; }
  [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }

  // Throws invalid_argument for different model or mode provenance and
  // overflow_error if any candidate moment or the sample count is not
  // representable. A failed observation leaves the accumulator unchanged.
  void observe(const ContinuousParticleModes &values);
  // Throws logic_error when no sample has been observed and overflow_error if
  // a normalized result is not finite.
  [[nodiscard]] ContinuousMatsubaraDensityCorrelations finish() const;

private:
  Model model_;
  MatsubaraModeSet modes_;
  std::size_t sample_count_ = 0;
  std::vector<std::complex<double>> amplitude_sums_;
  std::vector<double> centered_norm_sums_;
  std::vector<std::complex<double>> analytic_means_;
};

// Gauge response to the dimensionless positive-bond Peierls source. The full
// flux response R=<I I^*>/(beta*V) includes the event contact term and has
// units of energy per site. The sampled uncentred mean flux is retained as a
// time-reversal diagnostic; the present source-free model has exact mean zero.
class HoppingResponse {
public:
  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept { return modes_; }
  [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }
  // All component accessors check frequency, momentum, and physical-axis
  // indices and throw out_of_range for an unknown component.
  [[nodiscard]] std::complex<double> mean_flux(std::size_t frequency, std::size_t momentum,
                                               std::size_t axis) const;
  [[nodiscard]] std::complex<double> flux_response(std::size_t frequency, std::size_t momentum,
                                                   std::size_t left, std::size_t right) const;
  // <K_axis>/(beta*V), equal to the negative axis kinetic energy per site.
  [[nodiscard]] double diamagnetic(std::size_t axis) const;
  // The time-ordered paramagnetic current correlation D*delta-R.
  [[nodiscard]] std::complex<double> paramagnetic(std::size_t frequency, std::size_t momentum,
                                                  std::size_t left, std::size_t right) const;

private:
  friend class HoppingResponseAccumulator;

  HoppingResponse(Model model, MatsubaraModeSet modes,
                  std::vector<std::complex<double>> mean_flux_values,
                  std::vector<std::complex<double>> flux_response_values,
                  std::vector<double> diamagnetic_values, std::size_t sample_count);

  [[nodiscard]] std::size_t flux_index(std::size_t frequency, std::size_t momentum,
                                       std::size_t axis) const;
  [[nodiscard]] std::size_t response_index(std::size_t frequency, std::size_t momentum,
                                           std::size_t left, std::size_t right) const;

  Model model_;
  MatsubaraModeSet modes_;
  std::vector<std::complex<double>> mean_flux_values_;
  std::vector<std::complex<double>> flux_response_values_;
  std::vector<double> diamagnetic_values_;
  std::size_t sample_count_;
};

// Averages exact signed hopping-flux amplitudes and event counts for one
// complete free Model and selected Matsubara mode set. The exact source-free
// mean flux is zero, so response products are analytically centred. observe()
// validates every candidate sum and count before mutating the accumulator.
class HoppingResponseAccumulator {
public:
  // Throws invalid_argument when model beta/layout and modes differ.
  HoppingResponseAccumulator(Model model, MatsubaraModeSet modes);

  [[nodiscard]] const Model &model() const noexcept { return model_; }
  [[nodiscard]] const MatsubaraModeSet &modes() const noexcept { return modes_; }
  [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }

  // Throws invalid_argument for different model or mode provenance and
  // overflow_error if any candidate moment, event count, or sample count is
  // not representable. A failed observation leaves the accumulator unchanged.
  void observe(const ContinuousParticleModes &values);
  // Throws logic_error when no sample has been observed and overflow_error if
  // a normalized response is not finite.
  [[nodiscard]] HoppingResponse finish() const;

private:
  Model model_;
  MatsubaraModeSet modes_;
  std::size_t sample_count_ = 0;
  std::vector<std::complex<double>> flux_sums_;
  std::vector<std::complex<double>> response_sums_;
  std::vector<std::size_t> axis_event_count_sums_;
};

// Projects exact residence integrals and event impulses in one grouped event
// sweep. Throws invalid_argument when the context and plan geometry differ and
// overflow_error/length_error when an amplitude or result shape is not finite
// and representable.
[[nodiscard]] ContinuousParticleModes
continuous_particle_modes(const ContinuousMeasurementContext &context,
                          const ContinuousMatsubaraPlan &plan);

// One-off convenience overload that first owns the configuration's event
// geometry in a ContinuousMeasurementContext.
[[nodiscard]] ContinuousParticleModes
continuous_particle_modes(const ContinuousConfiguration &configuration,
                          const ContinuousMatsubaraPlan &plan);

} // namespace qmc

#endif
