#include "lattice_transform_detail.hpp"
#include "qmc/observables.hpp"
#include "qmc/torus_layout.hpp"

#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace qmc {

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
  std::vector<std::complex<double>> spatial_transform(grid_size);
  std::vector<std::size_t> displacement_components(layout.dimension());
  for (std::size_t time = 0; time < time_points; ++time) {
    for (std::size_t momentum = 0; momentum < volume; ++momentum) {
      const auto momentum_components = layout.decode(SiteId(momentum));
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

  MatsubaraDensityCorrelations result{
      .frequencies = std::vector<double>(time_points),
      .momentum_points = volume,
      .values = std::vector<std::complex<double>>(grid_size),
  };
  const double time_step = grid.beta() / static_cast<double>(time_points);
  for (std::size_t frequency = 0; frequency < time_points; ++frequency) {
    result.frequencies[frequency] =
        2.0 * std::numbers::pi * static_cast<double>(frequency) / grid.beta();
    for (std::size_t momentum = 0; momentum < volume; ++momentum) {
      std::complex<double> value{0.0, 0.0};
      for (std::size_t time = 0; time < time_points; ++time) {
        const double phase = 2.0 * std::numbers::pi * static_cast<double>(frequency * time) /
                             static_cast<double>(time_points);
        value += spatial_transform[(time * volume) + momentum] * std::polar(1.0, phase);
      }
      result.values[(frequency * volume) + momentum] = time_step * value;
    }
  }
  return result;
}

} // namespace qmc
