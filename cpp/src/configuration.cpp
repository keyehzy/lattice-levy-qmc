#include "qmc/configuration.hpp"

#include "checked_math.hpp"
#include "qmc/torus_layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc {
namespace {

enum class ValidationBoundary : std::uint8_t { Construction, Audit };

std::size_t dense_worldline_extent(const std::size_t particle_count, const std::size_t point_count,
                                   const std::size_t dimensions) {
  if (particle_count > std::numeric_limits<ParticleId>::max()) {
    throw std::invalid_argument("dense world-line particle count exceeds the ParticleId range");
  }
  if (dimensions == 0) {
    throw std::invalid_argument("dense world-line dimension must be positive");
  }
  const auto particle_points =
      detail::checked_product(particle_count, point_count, "dense world-line shape exceeds size_t");
  return detail::checked_product(particle_points, dimensions,
                                 "dense world-line shape exceeds size_t");
}

[[noreturn]] void fail_validation(const ValidationBoundary boundary, const char *message) {
  if (boundary == ValidationBoundary::Construction) {
    throw std::invalid_argument(message);
  }
  throw std::logic_error(message);
}

Coord winding_component(const Coord start, const Coord end, const Coord linear_size,
                        const ValidationBoundary boundary) {
  const auto [nonnegative, magnitude] = detail::displacement(start, end);
  const auto size = static_cast<std::uint64_t>(linear_size);
  if (magnitude % size != 0) {
    fail_validation(boundary, "cycle covering displacement is not a torus winding");
  }

  const std::uint64_t quotient = magnitude / size;
  const auto maximum = static_cast<std::uint64_t>(std::numeric_limits<Coord>::max());
  if (nonnegative) {
    if (quotient > maximum) {
      throw std::overflow_error("cycle winding exceeds int64 range");
    }
    return static_cast<Coord>(quotient);
  }

  const std::uint64_t minimum_magnitude = maximum + 1;
  if (quotient > minimum_magnitude) {
    throw std::overflow_error("cycle winding exceeds int64 range");
  }
  if (quotient == minimum_magnitude) {
    return std::numeric_limits<Coord>::min();
  }
  return -static_cast<Coord>(quotient);
}

Site derive_cycle_winding(const IdealBosonConfiguration &configuration,
                          const std::size_t cycle_index, const ValidationBoundary boundary) {
  const auto cycles = configuration.topology().cycles();
  if (cycle_index >= cycles.size()) {
    throw std::out_of_range("cycle index is out of range");
  }
  const Cycle &cycle = cycles[cycle_index];
  if (cycle.empty()) {
    fail_validation(boundary, "retained topology contains an empty cycle");
  }

  const Model &model = configuration.model();
  const DenseWorldlines &worldlines = configuration.covering_worldlines();
  const ParticleId root = cycle.front();
  const ParticleId last = cycle.back();
  Site winding(model.dimension());
  for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
    winding[axis] =
        winding_component(worldlines.at(root, 0, axis),
                          worldlines.at(last, configuration.time_links_per_beta(), axis),
                          model.linear_size(), boundary);
  }
  return winding;
}

void validate_configuration(const IdealBosonConfiguration &configuration,
                            const ValidationBoundary boundary) {
  const Model &model = configuration.model();
  const std::size_t time_links = configuration.time_links_per_beta();
  if (time_links < 1) {
    fail_validation(boundary, "time_links_per_beta must be positive");
  }
  const auto expected_points =
      detail::checked_add_size(time_links, 1, "world-line time-point count exceeds size_t");
  if (configuration.topology().size() != model.particle_count()) {
    fail_validation(boundary, "retained topology size does not match particle_count");
  }

  const DenseWorldlines &worldlines = configuration.covering_worldlines();
  if (worldlines.particle_count() != model.particle_count() ||
      worldlines.time_points() != expected_points || worldlines.dimension() != model.dimension()) {
    fail_validation(boundary,
                    "covering world-line shape does not match the retained configuration");
  }

  static_cast<void>(TorusLayout(model.linear_size(), model.dimension()));

  const auto cycles = configuration.topology().cycles();
  for (std::size_t cycle_index = 0; cycle_index < cycles.size(); ++cycle_index) {
    const Cycle &cycle = cycles[cycle_index];
    for (std::size_t index = 0; index + 1 < cycle.size(); ++index) {
      if (!std::ranges::equal(worldlines.site(cycle[index], time_links),
                              worldlines.site(cycle[index + 1], 0))) {
        fail_validation(boundary,
                        "covering world-line pieces do not join their permutation successor");
      }
    }
    static_cast<void>(derive_cycle_winding(configuration, cycle_index, boundary));
  }
}

} // namespace

DenseWorldlines::DenseWorldlines(const std::size_t particle_count, const std::size_t point_count,
                                 const std::size_t dimensions)
    : DenseWorldlines(
          particle_count, point_count, dimensions,
          std::vector<Coord>(dense_worldline_extent(particle_count, point_count, dimensions))) {}

DenseWorldlines::DenseWorldlines(const std::size_t particle_count, const std::size_t point_count,
                                 const std::size_t dimensions, std::vector<Coord> values)
    : particle_count_(particle_count), time_points_(point_count), dimension_(dimensions),
      values_(std::move(values)) {
  const auto expected = dense_worldline_extent(particle_count_, time_points_, dimension_);
  if (values_.size() != expected) {
    throw std::invalid_argument("dense world-line buffer does not match its declared shape");
  }
}

std::size_t DenseWorldlines::site_offset(const ParticleId particle,
                                         const std::size_t time_index) const {
  if (static_cast<std::size_t>(particle) >= particle_count_ || time_index >= time_points_) {
    throw std::out_of_range("dense world-line index is out of range");
  }
  return ((static_cast<std::size_t>(particle) * time_points_) + time_index) * dimension_;
}

Coord &DenseWorldlines::at(const ParticleId particle, const std::size_t time_index,
                           const std::size_t axis) {
  return const_cast<Coord &>(std::as_const(*this).at(particle, time_index, axis));
}

const Coord &DenseWorldlines::at(const ParticleId particle, const std::size_t time_index,
                                 const std::size_t axis) const {
  if (axis >= dimension_) {
    throw std::out_of_range("dense world-line axis is out of range");
  }
  return values_[site_offset(particle, time_index) + axis];
}

std::span<const Coord> DenseWorldlines::site(const ParticleId particle,
                                             const std::size_t time_index) const {
  const auto offset = site_offset(particle, time_index);
  return std::span<const Coord>(values_).subspan(offset, dimension_);
}

IdealBosonConfiguration::IdealBosonConfiguration(Model model, const std::size_t time_links_per_beta,
                                                 Permutation topology,
                                                 DenseWorldlines covering_worldlines)
    : model_(model), time_links_per_beta_(time_links_per_beta), topology_(std::move(topology)),
      covering_worldlines_(std::move(covering_worldlines)) {
  validate_configuration(*this, ValidationBoundary::Construction);
}

Site IdealBosonConfiguration::cycle_winding(const std::size_t cycle_index) const {
  return derive_cycle_winding(*this, cycle_index, ValidationBoundary::Audit);
}

void IdealBosonConfiguration::validate() const {
  validate_configuration(*this, ValidationBoundary::Audit);
}

namespace {

IdealBosonConfiguration
sample_ideal_boson_configuration_with_kernels(const CanonicalEnsemble &ensemble,
                                              const std::size_t time_links_per_beta, Random &random,
                                              const FreePathKernels &kernels) {
  const Model &model = ensemble.model();
  if (time_links_per_beta < 1) {
    throw std::invalid_argument("time_links_per_beta must be positive");
  }
  const auto time_points = detail::checked_add_size(time_links_per_beta, 1,
                                                    "world-line time-point count exceeds size_t");

  const std::vector<Cycle> cycle_labels = ensemble.sample_cycles(random);
  std::vector<ParticleId> successors(model.particle_count());
  DenseWorldlines covering_worldlines(model.particle_count(), time_points, model.dimension());

  for (const Cycle &labels : cycle_labels) {
    const auto length = labels.size();
    const double duration = static_cast<double>(length) * model.beta();
    if (!std::isfinite(duration)) {
      throw std::overflow_error("cycle duration overflowed");
    }

    Site base(model.dimension());
    Site winding(model.dimension());
    Site endpoint(model.dimension());
    for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
      base[axis] =
          static_cast<Coord>(random.uniform_index(static_cast<std::uint64_t>(model.linear_size())));
      winding[axis] = kernels.sample_winding_1d(duration, random);
      endpoint[axis] =
          detail::checked_add(base[axis],
                              detail::checked_scale(model.linear_size(), winding[axis],
                                                    "winding displacement exceeds int64 range"),
                              "covering endpoint exceeds int64 range");
    }

    const auto steps = detail::checked_product(length, time_links_per_beta,
                                               "cycle skeleton length exceeds size_t");
    const CoveringPath covering =
        kernels.sample_bridge_covering(base, endpoint, duration, steps, random);

    for (std::size_t cycle_index = 0; cycle_index < length; ++cycle_index) {
      const ParticleId label = labels[cycle_index];
      const auto start = cycle_index * time_links_per_beta;
      for (std::size_t time = 0; time <= time_links_per_beta; ++time) {
        for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
          covering_worldlines.at(label, time, axis) = covering[start + time][axis];
        }
      }
      successors[label] = labels[(cycle_index + 1) % length];
    }
  }

  return {model, time_links_per_beta, Permutation(std::move(successors)),
          std::move(covering_worldlines)};
}

} // namespace

IdealBosonConfiguration sample_ideal_boson_configuration(const CanonicalEnsemble &ensemble,
                                                         const std::size_t time_links_per_beta,
                                                         Random &random) {
  return sample_ideal_boson_configuration_with_kernels(ensemble, time_links_per_beta, random,
                                                       ensemble.free_path_kernels());
}

IdealBosonConfiguration sample_ideal_boson_configuration(const CanonicalEnsemble &ensemble,
                                                         const std::size_t time_links_per_beta,
                                                         Random &random,
                                                         const NumericalOptions &options) {
  return sample_ideal_boson_configuration_with_kernels(ensemble, time_links_per_beta, random,
                                                       FreePathKernels(ensemble.model(), options));
}

IdealBosonConfiguration sample_ideal_boson_configuration(const Model &model,
                                                         const std::size_t time_links_per_beta,
                                                         Random &random,
                                                         const NumericalOptions &options) {
  return sample_ideal_boson_configuration(CanonicalEnsemble(model, options), time_links_per_beta,
                                          random);
}

} // namespace qmc
