#include "qmc/configuration.hpp"

#include "qmc/torus_layout.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qmc {
namespace {

std::size_t checked_product(const std::size_t left, const std::size_t right,
                            const char *description) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(description);
  }
  return left * right;
}

Coord checked_scale(const Coord scale, const Coord value) {
  if (scale < 0) {
    throw std::logic_error("coordinate scale must be nonnegative");
  }
  if (scale == 0) {
    return 0;
  }
  if (value > 0 && scale > std::numeric_limits<Coord>::max() / value) {
    throw std::overflow_error("winding displacement exceeds int64 range");
  }
  if (value < 0 && value < std::numeric_limits<Coord>::min() / scale) {
    throw std::overflow_error("winding displacement exceeds int64 range");
  }
  return scale * value;
}

Coord checked_add(const Coord left, const Coord right) {
  if (right > 0 && left > std::numeric_limits<Coord>::max() - right) {
    throw std::overflow_error("covering endpoint exceeds int64 range");
  }
  if (right < 0 && left < std::numeric_limits<Coord>::min() - right) {
    throw std::overflow_error("covering endpoint exceeds int64 range");
  }
  return left + right;
}

void validate_site(const Site &site, const std::size_t dimension, const char *description) {
  if (site.size() != dimension) {
    throw std::logic_error(description);
  }
}

void validate_dense_buffers(const IdealBosonConfiguration &configuration) {
  configuration.worldlines.validate_shape();
  configuration.worldlines_covering.validate_shape();
  const auto expected_points = configuration.time_links_per_beta + 1;
  const auto has_expected_shape = [&configuration, expected_points](const DenseWorldlines &dense) {
    return dense.particles == configuration.model.particle_count &&
           dense.time_points == expected_points && dense.dimension == configuration.model.dimension;
  };
  if (!has_expected_shape(configuration.worldlines) ||
      !has_expected_shape(configuration.worldlines_covering)) {
    throw std::logic_error("dense world-line shape does not match the configuration model");
  }
}

void validate_permutation(const IdealBosonConfiguration &configuration) {
  if (configuration.permutation.size() != configuration.model.particle_count) {
    throw std::logic_error("permutation size does not match particle_count");
  }
  std::vector<bool> seen(configuration.model.particle_count, false);
  for (const ParticleId successor : configuration.permutation) {
    if (static_cast<std::size_t>(successor) >= configuration.model.particle_count ||
        seen[successor]) {
      throw std::logic_error("permutation is not a bijection");
    }
    seen[successor] = true;
  }
}

std::size_t validate_cycle_geometry(const IdealBosonConfiguration &configuration,
                                    const IdealCyclePath &cycle, const TorusLayout &layout) {
  if (cycle.labels.empty()) {
    throw std::logic_error("cycles must not be empty");
  }
  validate_site(cycle.base_point, configuration.model.dimension,
                "cycle base point has wrong dimension");
  validate_site(cycle.winding, configuration.model.dimension, "cycle winding has wrong dimension");
  for (const Coord coordinate : cycle.base_point) {
    if (coordinate < 0 || coordinate >= configuration.model.linear_size) {
      throw std::logic_error("cycle base point lies outside the torus");
    }
  }

  const auto steps = checked_product(cycle.labels.size(), configuration.time_links_per_beta,
                                     "cycle skeleton length exceeds size_t");
  if (steps == std::numeric_limits<std::size_t>::max() || cycle.covering_path.size() != steps + 1 ||
      cycle.torus_path.size() != steps + 1) {
    throw std::logic_error("cycle path length is inconsistent with its labels");
  }

  Site reduced(layout.dimension());
  for (std::size_t point = 0; point <= steps; ++point) {
    validate_site(cycle.covering_path[point], configuration.model.dimension,
                  "covering path site has wrong dimension");
    validate_site(cycle.torus_path[point], configuration.model.dimension,
                  "torus path site has wrong dimension");
    layout.reduce_into(cycle.covering_path[point], reduced);
    if (cycle.torus_path[point] != reduced) {
      throw std::logic_error("torus path is not the reduction of its covering path");
    }
  }
  for (std::size_t axis = 0; axis < configuration.model.dimension; ++axis) {
    const Coord displacement = checked_scale(configuration.model.linear_size, cycle.winding[axis]);
    if (checked_add(cycle.covering_path.front()[axis], displacement) !=
        cycle.covering_path.back()[axis]) {
      throw std::logic_error("cycle covering displacement does not match its winding");
    }
  }
  return steps;
}

void validate_cycle_assignment(const IdealBosonConfiguration &configuration,
                               const IdealCyclePath &cycle, std::vector<bool> &labels_seen) {
  for (std::size_t cycle_index = 0; cycle_index < cycle.labels.size(); ++cycle_index) {
    const ParticleId label = cycle.labels[cycle_index];
    if (static_cast<std::size_t>(label) >= configuration.model.particle_count ||
        labels_seen[label]) {
      throw std::logic_error("cycles do not partition the particle labels");
    }
    labels_seen[label] = true;
    const ParticleId successor = cycle.labels[(cycle_index + 1) % cycle.labels.size()];
    if (configuration.permutation[label] != successor) {
      throw std::logic_error("cycle order and permutation disagree");
    }

    const auto cycle_start = cycle_index * configuration.time_links_per_beta;
    for (std::size_t time = 0; time <= configuration.time_links_per_beta; ++time) {
      for (std::size_t axis = 0; axis < configuration.model.dimension; ++axis) {
        if (configuration.worldlines.at(label, time, axis) !=
                cycle.torus_path[cycle_start + time][axis] ||
            configuration.worldlines_covering.at(label, time, axis) !=
                cycle.covering_path[cycle_start + time][axis]) {
          throw std::logic_error("dense world lines do not match their cycle paths");
        }
      }
    }
  }
}

void validate_endpoint_joining(const IdealBosonConfiguration &configuration) {
  for (std::size_t particle = 0; particle < configuration.model.particle_count; ++particle) {
    const auto label = static_cast<ParticleId>(particle);
    const auto successor = configuration.permutation[particle];
    for (std::size_t axis = 0; axis < configuration.model.dimension; ++axis) {
      if (configuration.worldlines.at(label, configuration.time_links_per_beta, axis) !=
          configuration.worldlines.at(successor, 0, axis)) {
        throw std::logic_error("world-line endpoint does not join its permutation successor");
      }
    }
  }
}

} // namespace

DenseWorldlines::DenseWorldlines(const std::size_t particle_count, const std::size_t point_count,
                                 const std::size_t dimensions)
    : particles(particle_count), time_points(point_count), dimension(dimensions) {
  const auto particle_points =
      checked_product(particles, time_points, "dense world-line shape exceeds size_t");
  values.resize(
      checked_product(particle_points, dimension, "dense world-line shape exceeds size_t"));
}

Coord &DenseWorldlines::at(const ParticleId particle, const std::size_t time_index,
                           const std::size_t axis) {
  return const_cast<Coord &>(std::as_const(*this).at(particle, time_index, axis));
}

const Coord &DenseWorldlines::at(const ParticleId particle, const std::size_t time_index,
                                 const std::size_t axis) const {
  if (static_cast<std::size_t>(particle) >= particles || time_index >= time_points ||
      axis >= dimension) {
    throw std::out_of_range("dense world-line index is out of range");
  }
  validate_shape();
  const auto offset =
      ((static_cast<std::size_t>(particle) * time_points + time_index) * dimension) + axis;
  return values[offset];
}

void DenseWorldlines::validate_shape() const {
  const auto particle_points =
      checked_product(particles, time_points, "dense world-line shape exceeds size_t");
  const auto expected =
      checked_product(particle_points, dimension, "dense world-line shape exceeds size_t");
  if (values.size() != expected) {
    throw std::logic_error("dense world-line buffer does not match its declared shape");
  }
}

void IdealBosonConfiguration::validate() const {
  model.validate();
  if (time_links_per_beta < 1) {
    throw std::logic_error("time_links_per_beta must be positive");
  }
  if (!std::isfinite(log_ZN)) {
    throw std::logic_error("log_ZN must be finite");
  }
  if (time_links_per_beta == std::numeric_limits<std::size_t>::max()) {
    throw std::logic_error("world-line time-point count exceeds size_t");
  }

  validate_dense_buffers(*this);
  validate_permutation(*this);
  const TorusLayout layout(model.linear_size, model.dimension);
  std::vector<bool> labels_seen(model.particle_count, false);
  for (const IdealCyclePath &cycle : cycles) {
    static_cast<void>(validate_cycle_geometry(*this, cycle, layout));
    validate_cycle_assignment(*this, cycle, labels_seen);
  }
  if (std::ranges::find(labels_seen, false) != labels_seen.end()) {
    throw std::logic_error("cycles do not cover every particle label");
  }

  validate_endpoint_joining(*this);
}

IdealBosonConfiguration sample_ideal_boson_configuration(const CanonicalEnsemble &ensemble,
                                                         const std::size_t time_links_per_beta,
                                                         Random &random,
                                                         const NumericalOptions &options) {
  const Model &model = ensemble.model();
  const TorusLayout layout(model.linear_size, model.dimension);
  options.validate();
  if (time_links_per_beta < 1) {
    throw std::invalid_argument("time_links_per_beta must be positive");
  }
  if (time_links_per_beta == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("world-line time-point count exceeds size_t");
  }

  const std::vector<Cycle> cycle_labels = ensemble.sample_cycles(random);

  IdealBosonConfiguration configuration{
      .model = model,
      .time_links_per_beta = time_links_per_beta,
      .cycles = {},
      .permutation = std::vector<ParticleId>(model.particle_count),
      .worldlines = DenseWorldlines(model.particle_count, time_links_per_beta + 1, model.dimension),
      .worldlines_covering =
          DenseWorldlines(model.particle_count, time_links_per_beta + 1, model.dimension),
      .log_ZN = ensemble.log_partition(model.particle_count),
  };
  configuration.cycles.reserve(cycle_labels.size());

  for (const Cycle &labels : cycle_labels) {
    const auto length = labels.size();
    const double duration = static_cast<double>(length) * model.beta;
    if (!std::isfinite(duration)) {
      throw std::overflow_error("cycle duration overflowed");
    }

    Site base(model.dimension);
    Site winding(model.dimension);
    Site endpoint(model.dimension);
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      base[axis] =
          static_cast<Coord>(random.uniform_index(static_cast<std::uint64_t>(model.linear_size)));
      winding[axis] =
          sample_winding_1d(model.linear_size, duration, model.hopping, random, options);
      endpoint[axis] = checked_add(base[axis], checked_scale(model.linear_size, winding[axis]));
    }

    const auto steps =
        checked_product(length, time_links_per_beta, "cycle skeleton length exceeds size_t");
    CoveringPath covering =
        sample_bridge_covering(base, endpoint, duration, steps, model.hopping, random, options);
    CoveringPath torus(covering.size(), Site(model.dimension));
    for (std::size_t point = 0; point < covering.size(); ++point) {
      layout.reduce_into(covering[point], torus[point]);
    }

    for (std::size_t cycle_index = 0; cycle_index < length; ++cycle_index) {
      const ParticleId label = labels[cycle_index];
      const auto start = cycle_index * time_links_per_beta;
      for (std::size_t time = 0; time <= time_links_per_beta; ++time) {
        for (std::size_t axis = 0; axis < model.dimension; ++axis) {
          configuration.worldlines.at(label, time, axis) = torus[start + time][axis];
          configuration.worldlines_covering.at(label, time, axis) = covering[start + time][axis];
        }
      }
      configuration.permutation[label] = labels[(cycle_index + 1) % length];
    }

    configuration.cycles.push_back(IdealCyclePath{
        .labels = labels,
        .base_point = std::move(base),
        .winding = std::move(winding),
        .covering_path = std::move(covering),
        .torus_path = std::move(torus),
    });
  }

  configuration.validate();
  return configuration;
}

IdealBosonConfiguration sample_ideal_boson_configuration(const Model &model,
                                                         const std::size_t time_links_per_beta,
                                                         Random &random,
                                                         const NumericalOptions &options) {
  return sample_ideal_boson_configuration(CanonicalEnsemble(model), time_links_per_beta, random,
                                          options);
}

} // namespace qmc
