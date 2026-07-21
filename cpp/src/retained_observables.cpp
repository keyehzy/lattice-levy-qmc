#include "checked_math.hpp"
#include "lattice_transform_detail.hpp"
#include "qmc/observables.hpp"
#include "qmc/torus_layout.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc {
namespace {

void require_matching_measurement_context(const RetainedGrid &grid,
                                          const std::size_t particle_count,
                                          const RetainedMeasurementContext &context) {
  if (context.grid() != grid) {
    throw std::invalid_argument("retained measurement context has a different grid");
  }
  if (context.particle_count() != particle_count) {
    throw std::invalid_argument("retained measurement context has a different particle count");
  }
}

void require_matching_retained_configuration(const RetainedGrid &grid,
                                             const std::size_t particle_count,
                                             const IdealBosonConfiguration &configuration) {
  const Model &model = configuration.model();
  const RetainedGrid configuration_grid(model.beta(),
                                        TorusLayout(model.linear_size(), model.dimension()),
                                        configuration.time_links_per_beta());
  if (configuration_grid != grid) {
    throw std::invalid_argument("retained configuration has a different grid");
  }
  if (model.particle_count() != particle_count) {
    throw std::invalid_argument("retained configuration has a different particle count");
  }
}

std::size_t next_observation_count(const std::size_t sample_count) {
  return detail::checked_add_size(sample_count, 1, "observable sample count exceeds size_t");
}

void add_values(const std::span<double> sums, const std::span<const double> values) noexcept {
  assert(sums.size() == values.size());
  for (std::size_t index = 0; index < sums.size(); ++index) {
    sums[index] += values[index];
  }
}

void divide_values(const std::span<double> values, const double divisor) noexcept {
  for (double &value : values) {
    value /= divisor;
  }
}

double squared_norm(const std::span<const Coord> values) noexcept {
  double result = 0.0;
  for (const Coord component : values) {
    const auto value = static_cast<double>(component);
    result += value * value;
  }
  return result;
}

void accumulate_pair_counts(const std::span<const SiteId> positions, const TorusLayout &layout,
                            std::vector<double> &pair_counts) {
  for (const SiteId origin : positions) {
    for (const SiteId target : positions) {
      ++pair_counts[layout.flat_displacement(origin, target).value()];
    }
  }
}

void accumulate_structure_factor(const std::span<const SiteId> positions, const TorusLayout &layout,
                                 const std::size_t particle_count,
                                 std::vector<double> &structure_factor) {
  if (particle_count == 0) {
    return;
  }
  std::vector<std::size_t> site_components(layout.dimension());
  for (std::size_t momentum = 0; momentum < structure_factor.size(); ++momentum) {
    const auto momentum_components = layout.decode(SiteId(momentum));
    std::complex<double> density_mode{0.0, 0.0};
    for (const SiteId site : positions) {
      layout.decode_into(site, site_components);
      const double phase = detail::phase_for_indices(momentum_components, site_components, layout);
      density_mode += std::polar(1.0, -phase);
    }
    structure_factor[momentum] += std::norm(density_mode) / static_cast<double>(particle_count);
  }
}

} // namespace

std::vector<std::size_t> sampled_cycle_histogram(const IdealBosonConfiguration &configuration) {
  std::vector<std::size_t> result(configuration.model().particle_count() + 1);
  for (const Cycle &cycle : configuration.topology().cycles()) {
    ++result[cycle.size()];
  }
  return result;
}

std::size_t longest_cycle_length(const IdealBosonConfiguration &configuration) {
  std::size_t result = 0;
  for (const Cycle &cycle : configuration.topology().cycles()) {
    result = std::max(result, cycle.size());
  }
  return result;
}

Site total_winding(const IdealBosonConfiguration &configuration) {
  Site result(configuration.model().dimension());
  for (std::size_t cycle = 0; cycle < configuration.topology().cycles().size(); ++cycle) {
    const Site winding = configuration.cycle_winding(cycle);
    for (std::size_t axis = 0; axis < configuration.model().dimension(); ++axis) {
      const Coord value = winding[axis];
      if ((value > 0 && result[axis] > std::numeric_limits<Coord>::max() - value) ||
          (value < 0 && result[axis] < std::numeric_limits<Coord>::min() - value)) {
        throw std::overflow_error("total winding exceeds the coordinate range");
      }
      result[axis] += value;
    }
  }
  return result;
}

RetainedGrid::RetainedGrid(const double beta, TorusLayout layout, const std::size_t time_points)
    : beta_(beta), layout_(std::move(layout)), time_points_(time_points) {
  if (!std::isfinite(beta_) || beta_ < 0.0) {
    throw std::invalid_argument("retained-grid beta must be finite and nonnegative");
  }
  if (time_points_ == 0) {
    throw std::invalid_argument("retained-grid time point count must be positive");
  }
}

RetainedMeasurementContext::RetainedMeasurementContext(const IdealBosonConfiguration &configuration)
    : grid_(configuration.model().beta(),
            TorusLayout(configuration.model().linear_size(), configuration.model().dimension()),
            configuration.time_links_per_beta()),
      particle_count_(configuration.model().particle_count()) {
  const auto position_count = detail::checked_product(grid_.time_points(), particle_count_,
                                                      "retained-position grid exceeds size_t");
  positions_.reserve(position_count);
  const DenseWorldlines &worldlines = configuration.covering_worldlines();
  for (std::size_t time = 0; time < grid_.time_points(); ++time) {
    for (std::size_t particle = 0; particle < particle_count_; ++particle) {
      positions_.push_back(
          grid_.layout().encode_covering(worldlines.site(static_cast<ParticleId>(particle), time)));
    }
  }
}

std::span<const SiteId>
RetainedMeasurementContext::positions_at(const std::size_t time_index) const {
  if (time_index >= grid_.time_points()) {
    throw std::out_of_range("retained-position time index is outside the measurement grid");
  }
  return std::span<const SiteId>(positions_).subspan(time_index * particle_count_, particle_count_);
}

ImaginaryTimeDensityCorrelations::ImaginaryTimeDensityCorrelations(
    RetainedGrid grid, std::vector<double> connected_density)
    : grid_(std::move(grid)), connected_density_(std::move(connected_density)) {
  const auto expected_size = detail::checked_product(time_points(), spatial_points(),
                                                     "density-correlation grid exceeds size_t");
  if (connected_density_.size() != expected_size) {
    throw std::invalid_argument("density-correlation storage does not match its retained grid");
  }
}

double ImaginaryTimeDensityCorrelations::at(const std::size_t time_index,
                                            const SiteId displacement) const {
  if (time_index >= time_points() || displacement.value() >= spatial_points()) {
    throw std::out_of_range("density-correlation index is outside the retained grid");
  }
  return connected_density_[(time_index * spatial_points()) + displacement.value()];
}

EqualTimeObservables equal_time_observables(const RetainedMeasurementContext &context) {
  const TorusLayout &layout = context.grid().layout();
  const auto volume = layout.volume();
  const auto time_points = context.grid().time_points();
  const auto particle_count = context.particle_count();

  EqualTimeObservables result{
      .site_density = std::vector<double>(volume),
      .pair_correlation = std::vector<double>(volume),
      .static_structure_factor = std::vector<double>(volume),
      .onsite_occupation_probability = std::vector<double>(particle_count + 1),
  };
  std::vector<double> pair_counts(volume);
  std::vector<std::size_t> occupancies(volume);

  for (std::size_t time = 0; time < time_points; ++time) {
    std::ranges::fill(occupancies, 0);
    const auto positions = context.positions_at(time);
    for (const SiteId site : positions) {
      ++occupancies[site.value()];
    }
    for (std::size_t site = 0; site < volume; ++site) {
      const auto occupation = occupancies[site];
      result.site_density[site] += static_cast<double>(occupation);
      ++result.onsite_occupation_probability[occupation];
      const auto occupation_value = static_cast<double>(occupation);
      result.mean_occupation_squared += occupation_value * occupation_value;
      result.mean_factorial_occupation += occupation_value * std::max(0.0, occupation_value - 1.0);
    }
    accumulate_pair_counts(positions, layout, pair_counts);
    accumulate_structure_factor(positions, layout, particle_count, result.static_structure_factor);
  }

  const auto slice_count = static_cast<double>(time_points);
  const double site_observation_count = slice_count * static_cast<double>(volume);
  for (double &density : result.site_density) {
    density /= slice_count;
  }
  for (double &probability : result.onsite_occupation_probability) {
    probability /= site_observation_count;
  }
  result.mean_occupation_squared /= site_observation_count;
  result.mean_factorial_occupation /= site_observation_count;
  for (double &structure_factor : result.static_structure_factor) {
    structure_factor /= slice_count;
  }

  if (particle_count > 0) {
    const double density = static_cast<double>(particle_count) / static_cast<double>(volume);
    for (std::size_t displacement = 0; displacement < volume; ++displacement) {
      const double correlation = pair_counts[displacement] / site_observation_count;
      const double self_term = displacement == 0 ? density : 0.0;
      result.pair_correlation[displacement] = (correlation - self_term) / (density * density);
    }
  }
  return result;
}

EqualTimeObservables equal_time_observables(const IdealBosonConfiguration &configuration) {
  return equal_time_observables(RetainedMeasurementContext(configuration));
}

EqualTimeAccumulator::EqualTimeAccumulator(RetainedGrid grid, const std::size_t particle_count)
    : grid_(std::move(grid)), particle_count_(particle_count),
      sums_{
          .site_density = std::vector<double>(grid_.layout().volume()),
          .pair_correlation = std::vector<double>(grid_.layout().volume()),
          .static_structure_factor = std::vector<double>(grid_.layout().volume()),
          .onsite_occupation_probability = std::vector<double>(detail::checked_add_size(
              particle_count_, 1, "onsite-probability extent exceeds size_t")),
      } {}

void EqualTimeAccumulator::observe(const RetainedMeasurementContext &context) {
  require_matching_measurement_context(grid_, particle_count_, context);
  const auto updated_sample_count = next_observation_count(sample_count_);
  const auto observation = equal_time_observables(context);
  add_values(sums_.site_density, observation.site_density);
  add_values(sums_.pair_correlation, observation.pair_correlation);
  add_values(sums_.static_structure_factor, observation.static_structure_factor);
  add_values(sums_.onsite_occupation_probability, observation.onsite_occupation_probability);
  sums_.mean_occupation_squared += observation.mean_occupation_squared;
  sums_.mean_factorial_occupation += observation.mean_factorial_occupation;
  sample_count_ = updated_sample_count;
}

EqualTimeObservables EqualTimeAccumulator::finish() const {
  if (sample_count_ == 0) {
    throw std::logic_error("cannot finish an equal-time accumulator without samples");
  }
  EqualTimeObservables result = sums_;
  const auto divisor = static_cast<double>(sample_count_);
  divide_values(result.site_density, divisor);
  divide_values(result.pair_correlation, divisor);
  divide_values(result.static_structure_factor, divisor);
  divide_values(result.onsite_occupation_probability, divisor);
  result.mean_occupation_squared /= divisor;
  result.mean_factorial_occupation /= divisor;
  return result;
}

ImaginaryTimeDensityCorrelations
retained_density_correlations(const RetainedMeasurementContext &context) {
  const RetainedGrid &grid = context.grid();
  const TorusLayout &layout = grid.layout();
  const auto volume = layout.volume();
  const auto time_points = grid.time_points();
  const auto grid_size =
      detail::checked_product(time_points, volume, "density-correlation grid exceeds size_t");
  std::vector<double> connected_density(grid_size);

  for (std::size_t lag = 0; lag < time_points; ++lag) {
    for (std::size_t origin_time = 0; origin_time < time_points; ++origin_time) {
      const auto time_until_wrap = time_points - origin_time;
      const std::size_t target_time =
          lag >= time_until_wrap ? lag - time_until_wrap : origin_time + lag;
      const auto origins = context.positions_at(origin_time);
      const auto targets = context.positions_at(target_time);
      for (const SiteId origin : origins) {
        for (const SiteId target : targets) {
          const auto displacement = layout.flat_displacement(origin, target);
          ++connected_density[(lag * volume) + displacement.value()];
        }
      }
    }
  }

  const double normalization = static_cast<double>(time_points) * static_cast<double>(volume);
  const double density =
      static_cast<double>(context.particle_count()) / static_cast<double>(volume);
  for (double &correlation : connected_density) {
    correlation = (correlation / normalization) - (density * density);
  }
  return {grid, std::move(connected_density)};
}

ImaginaryTimeDensityCorrelations
retained_density_correlations(const IdealBosonConfiguration &configuration) {
  return retained_density_correlations(RetainedMeasurementContext(configuration));
}

RetainedDensityCorrelationAccumulator::RetainedDensityCorrelationAccumulator(
    RetainedGrid grid, const std::size_t particle_count)
    : grid_(std::move(grid)), particle_count_(particle_count),
      connected_density_sums_(
          detail::checked_product(grid_.time_points(), grid_.layout().volume(),
                                  "density-correlation accumulator grid exceeds size_t")) {}

void RetainedDensityCorrelationAccumulator::observe(const RetainedMeasurementContext &context) {
  require_matching_measurement_context(grid_, particle_count_, context);
  const auto updated_sample_count = next_observation_count(sample_count_);
  const auto observation = retained_density_correlations(context);
  add_values(connected_density_sums_, observation.connected_density());
  sample_count_ = updated_sample_count;
}

ImaginaryTimeDensityCorrelations RetainedDensityCorrelationAccumulator::finish() const {
  if (sample_count_ == 0) {
    throw std::logic_error("cannot finish a density-correlation accumulator without samples");
  }
  std::vector<double> result = connected_density_sums_;
  divide_values(result, static_cast<double>(sample_count_));
  return {grid_, std::move(result)};
}

RetainedGeometryObservables
retained_geometry_observables(const IdealBosonConfiguration &configuration) {
  const Model &model = configuration.model();
  const TorusLayout layout(model.linear_size(), model.dimension());
  const auto volume = layout.volume();
  const auto time_points = configuration.time_links_per_beta();
  const DenseWorldlines &worldlines = configuration.covering_worldlines();
  const auto grid_size =
      detail::checked_product(time_points, volume, "retained-geometry grid exceeds size_t");
  RetainedGeometryObservables result{
      .time_points = time_points,
      .displacement_points = volume,
      .mean_square_displacement = std::vector<double>(time_points),
      .return_probability = std::vector<double>(time_points),
      .displacement_probability = std::vector<double>(grid_size),
  };
  if (model.particle_count() == 0) {
    return result;
  }

  const auto particle_count = static_cast<double>(model.particle_count());
  for (std::size_t time = 0; time < time_points; ++time) {
    for (std::size_t particle = 0; particle < model.particle_count(); ++particle) {
      double squared_displacement = 0.0;
      const auto label = static_cast<ParticleId>(particle);
      for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
        const double value = static_cast<double>(worldlines.at(label, time, axis)) -
                             static_cast<double>(worldlines.at(label, 0, axis));
        squared_displacement += value * value;
      }
      result.mean_square_displacement[time] += squared_displacement / particle_count;
      const SiteId flat =
          layout.flat_displacement(layout.encode_covering(worldlines.site(label, 0)),
                                   layout.encode_covering(worldlines.site(label, time)));
      result.displacement_probability[(time * volume) + flat.value()] += 1.0 / particle_count;
      if (flat.value() == 0) {
        result.return_probability[time] += 1.0 / particle_count;
      }
    }
  }
  return result;
}

RetainedGeometryAccumulator::RetainedGeometryAccumulator(RetainedGrid grid,
                                                         const std::size_t particle_count)
    : grid_(std::move(grid)), particle_count_(particle_count),
      sums_{
          .time_points = grid_.time_points(),
          .displacement_points = grid_.layout().volume(),
          .mean_square_displacement = std::vector<double>(grid_.time_points()),
          .return_probability = std::vector<double>(grid_.time_points()),
          .displacement_probability = std::vector<double>(
              detail::checked_product(grid_.time_points(), grid_.layout().volume(),
                                      "retained-geometry accumulator grid exceeds size_t")),
      } {}

void RetainedGeometryAccumulator::observe(const IdealBosonConfiguration &configuration) {
  require_matching_retained_configuration(grid_, particle_count_, configuration);
  const auto updated_sample_count = next_observation_count(sample_count_);
  const auto observation = retained_geometry_observables(configuration);
  add_values(sums_.mean_square_displacement, observation.mean_square_displacement);
  add_values(sums_.return_probability, observation.return_probability);
  add_values(sums_.displacement_probability, observation.displacement_probability);
  sample_count_ = updated_sample_count;
}

RetainedGeometryObservables RetainedGeometryAccumulator::finish() const {
  if (sample_count_ == 0) {
    throw std::logic_error("cannot finish a retained-geometry accumulator without samples");
  }
  RetainedGeometryObservables result = sums_;
  const auto divisor = static_cast<double>(sample_count_);
  divide_values(result.mean_square_displacement, divisor);
  divide_values(result.return_probability, divisor);
  divide_values(result.displacement_probability, divisor);
  return result;
}

std::vector<RetainedCycleGeometry>
retained_cycle_geometry(const IdealBosonConfiguration &configuration) {
  const Model &model = configuration.model();
  const auto time_links = configuration.time_links_per_beta();
  const DenseWorldlines &worldlines = configuration.covering_worldlines();
  const auto cycles = configuration.topology().cycles();
  std::vector<RetainedCycleGeometry> result;
  result.reserve(cycles.size());
  for (std::size_t cycle_index = 0; cycle_index < cycles.size(); ++cycle_index) {
    const Cycle &cycle = cycles[cycle_index];
    const auto distinct_points = detail::checked_product(
        cycle.size(), time_links, "cycle retained-point count exceeds size_t");
    std::vector<double> center(model.dimension());
    for (const ParticleId label : cycle) {
      for (std::size_t time = 0; time < time_links; ++time) {
        for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
          center[axis] += static_cast<double>(worldlines.at(label, time, axis));
        }
      }
    }
    for (double &component : center) {
      component /= static_cast<double>(distinct_points);
    }

    RetainedCycleGeometry geometry{
        .length = cycle.size(),
        .winding = configuration.cycle_winding(cycle_index),
    };
    for (const ParticleId label : cycle) {
      for (std::size_t time = 0; time < time_links; ++time) {
        double squared_radius = 0.0;
        for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
          const double displacement =
              static_cast<double>(worldlines.at(label, time, axis)) - center[axis];
          squared_radius += displacement * displacement;
        }
        geometry.radius_of_gyration_squared +=
            squared_radius / static_cast<double>(distinct_points);
        geometry.maximum_radius_squared = std::max(geometry.maximum_radius_squared, squared_radius);
      }
    }
    result.push_back(std::move(geometry));
  }
  return result;
}

CycleStatisticsAccumulator::CycleStatisticsAccumulator(RetainedGrid grid,
                                                       const std::size_t particle_count)
    : grid_(std::move(grid)), particle_count_(particle_count),
      macroscopic_cycle_threshold_(
          particle_count_ == 0 ? 1 : (particle_count_ / 2) + (particle_count_ % 2)) {
  const auto extent =
      detail::checked_add_size(particle_count_, 1, "cycle-statistics extent exceeds size_t");
  sums_.mean_cycle_count.resize(extent);
  sums_.mean_particles.resize(extent);
  sums_.mean_cycle_winding_squared.resize(extent);
  sums_.mean_radius_of_gyration_squared.resize(extent);
  sums_.mean_maximum_radius_squared.resize(extent);
  sums_.longest_cycle_probability.resize(extent);
  cycle_occurrences_.resize(extent);
}

std::vector<RetainedCycleGeometry>
CycleStatisticsAccumulator::observe(const IdealBosonConfiguration &configuration) {
  require_matching_retained_configuration(grid_, particle_count_, configuration);
  const auto updated_sample_count = next_observation_count(sample_count_);
  auto geometry = retained_cycle_geometry(configuration);

  std::size_t longest_cycle = 0;
  std::size_t macroscopic_particles = 0;
  for (const RetainedCycleGeometry &cycle : geometry) {
    assert(cycle.length > 0 && cycle.length <= particle_count_);
    sums_.mean_cycle_count[cycle.length] += 1.0;
    sums_.mean_particles[cycle.length] += static_cast<double>(cycle.length);
    sums_.mean_cycle_winding_squared[cycle.length] += squared_norm(cycle.winding);
    sums_.mean_radius_of_gyration_squared[cycle.length] += cycle.radius_of_gyration_squared;
    sums_.mean_maximum_radius_squared[cycle.length] += cycle.maximum_radius_squared;
    cycle_occurrences_[cycle.length] += 1.0;
    longest_cycle = std::max(longest_cycle, cycle.length);
    if (cycle.length >= macroscopic_cycle_threshold_) {
      macroscopic_particles += cycle.length;
    }
  }
  sums_.longest_cycle_probability[longest_cycle] += 1.0;
  if (particle_count_ > 0) {
    sums_.macroscopic_cycle_fraction +=
        static_cast<double>(macroscopic_particles) / static_cast<double>(particle_count_);
  }
  sample_count_ = updated_sample_count;
  return geometry;
}

SampledCycleStatistics CycleStatisticsAccumulator::finish() const {
  if (sample_count_ == 0) {
    throw std::logic_error("cannot finish a cycle-statistics accumulator without samples");
  }
  SampledCycleStatistics result = sums_;
  const auto sample_divisor = static_cast<double>(sample_count_);
  divide_values(result.mean_cycle_count, sample_divisor);
  divide_values(result.mean_particles, sample_divisor);
  divide_values(result.longest_cycle_probability, sample_divisor);
  result.macroscopic_cycle_fraction /= sample_divisor;
  for (std::size_t length = 1; length <= particle_count_; ++length) {
    if (cycle_occurrences_[length] > 0.0) {
      result.mean_cycle_winding_squared[length] /= cycle_occurrences_[length];
      result.mean_radius_of_gyration_squared[length] /= cycle_occurrences_[length];
      result.mean_maximum_radius_squared[length] /= cycle_occurrences_[length];
    }
  }
  return result;
}

WindingAccumulator::WindingAccumulator(RetainedGrid grid, const std::size_t particle_count)
    : grid_(std::move(grid)), particle_count_(particle_count),
      sums_{
          .second_moment = std::vector<double>(grid_.layout().dimension()),
          .fourth_moment = std::vector<double>(grid_.layout().dimension()),
      } {}

Site WindingAccumulator::observe(const IdealBosonConfiguration &configuration) {
  require_matching_retained_configuration(grid_, particle_count_, configuration);
  const auto updated_sample_count = next_observation_count(sample_count_);
  Site winding = total_winding(configuration);
  bool nonzero = false;
  for (std::size_t axis = 0; axis < winding.size(); ++axis) {
    const auto component = static_cast<double>(winding[axis]);
    const double square = component * component;
    sums_.second_moment[axis] += square;
    sums_.fourth_moment[axis] += square * square;
    nonzero = nonzero || winding[axis] != 0;
  }
  if (nonzero) {
    sums_.nonzero_probability += 1.0;
  }
  sample_count_ = updated_sample_count;
  return winding;
}

WindingStatistics WindingAccumulator::finish() const {
  if (sample_count_ == 0) {
    throw std::logic_error("cannot finish a winding accumulator without samples");
  }
  WindingStatistics result = sums_;
  const auto divisor = static_cast<double>(sample_count_);
  divide_values(result.second_moment, divisor);
  divide_values(result.fourth_moment, divisor);
  result.nonzero_probability /= divisor;
  return result;
}

} // namespace qmc
