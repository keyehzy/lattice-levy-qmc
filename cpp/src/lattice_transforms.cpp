#include "lattice_transform_detail.hpp"
#include "qmc/observables.hpp"
#include "qmc/torus_layout.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc {
namespace {

MatsubaraModeRequest retained_mode_request(const RetainedGrid &grid) {
  if (grid.time_points() - 1 > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("retained Matsubara frequency index exceeds int64 range");
  }

  MatsubaraModeRequest request;
  if (grid.layout().volume() > request.momentum_indices.max_size()) {
    throw std::length_error("retained Matsubara momenta exceed vector capacity");
  }
  if (grid.time_points() > request.frequency_indices.max_size()) {
    throw std::length_error("retained Matsubara frequencies exceed vector capacity");
  }
  request.momentum_indices.reserve(grid.layout().volume());
  for (std::size_t momentum = 0; momentum < grid.layout().volume(); ++momentum) {
    request.momentum_indices.push_back(grid.layout().decode(SiteId(momentum)));
  }
  request.frequency_indices.reserve(grid.time_points());
  for (std::size_t frequency = 0; frequency < grid.time_points(); ++frequency) {
    request.frequency_indices.push_back(static_cast<std::int64_t>(frequency));
  }
  return request;
}

} // namespace

MatsubaraDensityCorrelations::MatsubaraDensityCorrelations(
    RetainedGrid source, MatsubaraModeField<std::complex<double>> values)
    : grid_(std::move(source)), values_(std::move(values)) {
  const MatsubaraModeSet &mode_set = values_.modes();
  if (mode_set.beta() != grid_.beta() || mode_set.layout() != grid_.layout()) {
    throw std::invalid_argument("retained Matsubara field geometry does not match its source grid");
  }
  if (grid_.time_points() - 1 >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("retained Matsubara frequency index exceeds int64 range");
  }
  if (mode_set.frequency_count() != grid_.time_points()) {
    throw std::invalid_argument("retained Matsubara frequencies do not match the source grid");
  }
  for (std::size_t frequency = 0; frequency < grid_.time_points(); ++frequency) {
    if (mode_set.frequency_index(frequency) != static_cast<std::int64_t>(frequency)) {
      throw std::invalid_argument(
          "retained Matsubara frequencies must be ordered from zero through M-1");
    }
  }
  if (mode_set.momentum_count() != grid_.layout().volume()) {
    throw std::invalid_argument("retained Matsubara momenta do not cover the source layout");
  }
  for (std::size_t momentum = 0; momentum < grid_.layout().volume(); ++momentum) {
    const std::vector<std::size_t> expected = grid_.layout().decode(SiteId(momentum));
    if (!std::ranges::equal(mode_set.momentum_indices(momentum), expected)) {
      throw std::invalid_argument("retained Matsubara momenta are not in flat torus order");
    }
  }
}

MatsubaraDensityCorrelations
retained_grid_matsubara_transform(const ImaginaryTimeDensityCorrelations &correlations) {
  const RetainedGrid &grid = correlations.grid();
  if (grid.beta() <= 0.0) {
    throw std::invalid_argument("Matsubara transform requires beta > 0");
  }
  const TorusLayout &layout = grid.layout();
  const auto volume = layout.volume();
  const auto time_points = grid.time_points();
  const auto connected_density = correlations.connected_density();
  const auto grid_size = connected_density.size();
  MatsubaraModeSet modes(grid.beta(), layout, retained_mode_request(grid));
  std::vector<std::complex<double>> spatial_transform(grid_size);
  std::vector<std::size_t> displacement_components(layout.dimension());
  for (std::size_t time = 0; time < time_points; ++time) {
    for (std::size_t momentum = 0; momentum < volume; ++momentum) {
      const auto momentum_components = modes.momentum_indices(momentum);
      std::complex<double> value{0.0, 0.0};
      for (std::size_t displacement = 0; displacement < volume; ++displacement) {
        layout.decode_into(SiteId(displacement), displacement_components);
        const double phase =
            detail::phase_for_indices(momentum_components, displacement_components, layout);
        value += connected_density[(time * volume) + displacement] * std::polar(1.0, -phase);
      }
      spatial_transform[(time * volume) + momentum] = value;
    }
  }

  std::vector<std::complex<double>> values(grid_size);
  const double time_step = grid.beta() / static_cast<double>(time_points);
  for (std::size_t frequency = 0; frequency < time_points; ++frequency) {
    for (std::size_t momentum = 0; momentum < volume; ++momentum) {
      std::complex<double> value{0.0, 0.0};
      for (std::size_t time = 0; time < time_points; ++time) {
        const double phase = 2.0 * std::numbers::pi * static_cast<double>(frequency) *
                             static_cast<double>(time) / static_cast<double>(time_points);
        value += spatial_transform[(time * volume) + momentum] * std::polar(1.0, phase);
      }
      values[(frequency * volume) + momentum] = time_step * value;
    }
  }
  return {grid, MatsubaraModeField<std::complex<double>>(std::move(modes), std::move(values))};
}

} // namespace qmc
