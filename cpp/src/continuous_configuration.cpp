#include "qmc/continuous_configuration.hpp"

#include "checked_math.hpp"
#include "continuous_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace qmc {
namespace {

bool nearly_equal_time(const double left, const double right) {
  if (left == right) {
    return true;
  }
  const double scale = std::max(std::abs(left), std::abs(right));
  return std::abs(left - right) <= 16.0 * std::numeric_limits<double>::epsilon() * scale;
}

void validate_continuous_model(const Model &model) {
  if (model.beta <= 0.0) {
    throw std::invalid_argument("continuous configurations require positive beta");
  }
}

void validate_permutation(const ContinuousConfiguration &configuration, const Model &model) {
  if (configuration.permutation.size() != model.particle_count) {
    throw std::logic_error("continuous permutation size does not match particle_count");
  }
  std::vector<bool> seen(model.particle_count, false);
  for (const ParticleId successor : configuration.permutation) {
    if (static_cast<std::size_t>(successor) >= model.particle_count || seen[successor]) {
      throw std::logic_error("continuous permutation is not a bijection");
    }
    seen[successor] = true;
  }
}

void validate_cycles(const ContinuousConfiguration &configuration, const Model &model) {
  std::vector<bool> labels_seen(model.particle_count, false);
  for (const Cycle &cycle : configuration.cycles) {
    if (cycle.empty()) {
      throw std::logic_error("continuous cycles must not be empty");
    }
    for (std::size_t index = 0; index < cycle.size(); ++index) {
      const ParticleId label = cycle[index];
      if (static_cast<std::size_t>(label) >= model.particle_count || labels_seen[label]) {
        throw std::logic_error("continuous cycles do not partition particle labels");
      }
      labels_seen[label] = true;
      if (configuration.permutation[label] != cycle[(index + 1) % cycle.size()]) {
        throw std::logic_error("continuous cycle order and permutation disagree");
      }
    }
  }
  if (std::ranges::find(labels_seen, false) != labels_seen.end()) {
    throw std::logic_error("continuous cycles do not cover every particle label");
  }
}

void validate_worldlines(const ContinuousConfiguration &configuration, const Model &model) {
  if (configuration.worldlines.size() != model.particle_count) {
    throw std::logic_error("continuous worldline count does not match particle_count");
  }
  for (std::size_t particle = 0; particle < model.particle_count; ++particle) {
    const ContinuousPath &path = configuration.worldlines[particle];
    path.validate(model.dimension);
    if (!nearly_equal_time(path.duration, model.beta)) {
      throw std::logic_error("continuous worldline duration does not match beta");
    }
    const ContinuousPath &successor = configuration.worldlines[configuration.permutation[particle]];
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      if (torus_mod(path.end[axis], model.linear_size) !=
          torus_mod(successor.start[axis], model.linear_size)) {
        throw std::logic_error(
            "continuous worldline endpoint does not join its permutation successor");
      }
    }
  }
}

} // namespace

namespace detail {

std::vector<ContinuousPath> sample_paths_for_cycle(const Cycle &labels, const Model &model,
                                                   Random &random,
                                                   const NumericalOptions &options) {
  validate_continuous_model(model);
  options.validate();
  if (labels.empty()) {
    throw std::invalid_argument("cannot sample paths for an empty cycle");
  }

  const double duration = static_cast<double>(labels.size()) * model.beta;
  if (!std::isfinite(duration)) {
    throw std::overflow_error("continuous cycle duration overflowed");
  }
  Site base(model.dimension);
  Site endpoint(model.dimension);
  for (std::size_t axis = 0; axis < model.dimension; ++axis) {
    base[axis] =
        static_cast<Coord>(random.uniform_index(static_cast<std::uint64_t>(model.linear_size)));
    const Coord winding =
        sample_winding_1d(model.linear_size, duration, model.hopping, random, options);
    const Coord winding_displacement = checked_scale(
        model.linear_size, winding, "continuous winding displacement exceeds int64 range");
    endpoint[axis] = checked_add(base[axis], winding_displacement,
                                 "continuous cycle endpoint exceeds int64 range");
  }

  ContinuousPath long_path =
      sample_continuous_bridge(base, endpoint, duration, model.hopping, random, options);
  std::vector<double> cuts;
  cuts.reserve(labels.size() - 1);
  for (std::size_t index = 1; index < labels.size(); ++index) {
    const double cut = static_cast<double>(index) * model.beta;
    if (!std::isfinite(cut)) {
      throw std::overflow_error("continuous cycle cut time overflowed");
    }
    cuts.push_back(cut);
  }
  auto paths = split_continuous_path(long_path, cuts);
  if (paths.size() != labels.size()) {
    throw std::logic_error("continuous cycle splitting returned the wrong number of paths");
  }
  // Floating-point products j*beta can make adjacent differences differ from
  // beta by a few ulps. Worldline time is exactly [0,beta], so normalize that
  // harmless arithmetic drift before the paths enter the Markov state.
  for (ContinuousPath &path : paths) {
    if (!nearly_equal_time(path.duration, model.beta)) {
      throw std::runtime_error("continuous cycle piece duration drift exceeds tolerance");
    }
    for (JumpEvent &event : path.events) {
      if (event.time > model.beta && nearly_equal_time(event.time, model.beta)) {
        event.time = model.beta;
      }
    }
    path.duration = model.beta;
    path.validate(model.dimension);
  }
  return paths;
}

} // namespace detail

std::vector<Cycle> cycles_from_permutation(const std::span<const ParticleId> permutation) {
  std::vector<bool> appeared(permutation.size(), false);
  for (const ParticleId successor : permutation) {
    const auto index = static_cast<std::size_t>(successor);
    if (index >= permutation.size() || appeared[index]) {
      throw std::invalid_argument("permutation is invalid");
    }
    appeared[index] = true;
  }

  std::vector<bool> seen(permutation.size(), false);
  std::vector<Cycle> cycles;
  for (std::size_t root = 0; root < permutation.size(); ++root) {
    if (seen[root]) {
      continue;
    }
    Cycle cycle;
    std::size_t current = root;
    while (!seen[current]) {
      seen[current] = true;
      cycle.push_back(static_cast<ParticleId>(current));
      current = static_cast<std::size_t>(permutation[current]);
    }
    if (current != root) {
      throw std::invalid_argument("permutation traversal did not close at its root");
    }
    cycles.push_back(std::move(cycle));
  }
  return cycles;
}

ContinuousConfiguration rotate_configuration_time_origin(const ContinuousConfiguration &state,
                                                         const Model &model, const double shift) {
  state.validate(model);
  if (!std::isfinite(shift) || shift < 0.0 || shift >= model.beta) {
    throw std::invalid_argument("time-origin shift must lie in [0, beta)");
  }
  if (shift == 0.0 || model.particle_count == 0) {
    return state;
  }

  std::vector<ContinuousPath> paths;
  paths.reserve(state.worldlines.size());
  for (std::size_t particle = 0; particle < state.worldlines.size(); ++particle) {
    const ContinuousPath &path = state.worldlines[particle];
    const ContinuousPath &successor = state.worldlines[state.permutation[particle]];
    const auto path_cut = std::ranges::lower_bound(path.events, shift, {}, &JumpEvent::time);
    const auto successor_cut =
        std::ranges::lower_bound(successor.events, shift, {}, &JumpEvent::time);

    // Use the left limit for the new start. A jump exactly at shift is retained
    // below at time zero, preserving right-continuous positions.
    Site start = path.start;
    for (auto event = path.events.begin(); event != path_cut; ++event) {
      start[event->axis] =
          detail::checked_add(start[event->axis], static_cast<Coord>(event->direction),
                              "rotated path start exceeds int64 range");
    }

    std::vector<JumpEvent> events;
    const auto suffix_count = static_cast<std::size_t>(path.events.end() - path_cut);
    const auto prefix_count = static_cast<std::size_t>(successor_cut - successor.events.begin());
    events.reserve(
        detail::checked_add_size(suffix_count, prefix_count, "rotated event count exceeds size_t"));
    for (auto event = path_cut; event != path.events.end(); ++event) {
      events.push_back(JumpEvent{
          .time = event->time - shift,
          .axis = event->axis,
          .direction = event->direction,
      });
    }
    for (auto event = successor.events.begin(); event != successor_cut; ++event) {
      events.push_back(JumpEvent{
          .time = (model.beta - shift) + event->time,
          .axis = event->axis,
          .direction = event->direction,
      });
    }

    Site end = successor.start;
    for (auto event = successor.events.begin(); event != successor_cut; ++event) {
      end[event->axis] = detail::checked_add(end[event->axis], static_cast<Coord>(event->direction),
                                             "rotated path endpoint exceeds int64 range");
    }
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      const Coord translation =
          detail::checked_subtract(path.end[axis], successor.start[axis],
                                   "time-rotation path translation exceeds int64 range");
      if (translation % model.linear_size != 0) {
        throw std::logic_error("invalid path connectivity during time rotation");
      }
      end[axis] =
          detail::checked_add(end[axis], translation, "rotated path endpoint exceeds int64 range");
    }

    ContinuousPath rotated_path{
        .duration = model.beta,
        .start = std::move(start),
        .end = std::move(end),
        .events = std::move(events),
    };
    rotated_path.validate(model.dimension);
    paths.push_back(std::move(rotated_path));
  }

  ContinuousConfiguration rotated{
      .cycles = state.cycles,
      .permutation = state.permutation,
      .worldlines = std::move(paths),
      .log_Z0_N = state.log_Z0_N,
  };
  rotated.validate(model);
  return rotated;
}

void ContinuousConfiguration::validate(const Model &model) const {
  validate_continuous_model(model);
  if (!std::isfinite(log_Z0_N)) {
    throw std::logic_error("continuous log_Z0_N must be finite");
  }
  validate_permutation(*this, model);
  validate_cycles(*this, model);
  validate_worldlines(*this, model);
}

std::size_t ContinuousConfiguration::event_count() const {
  std::size_t count = 0;
  for (const ContinuousPath &path : worldlines) {
    count = detail::checked_add_size(count, path.event_count(),
                                     "continuous configuration event count exceeds size_t");
  }
  return count;
}

std::vector<std::size_t> ContinuousConfiguration::cycle_lengths() const {
  std::vector<std::size_t> lengths;
  lengths.reserve(cycles.size());
  for (const Cycle &cycle : cycles) {
    lengths.push_back(cycle.size());
  }
  return lengths;
}

std::vector<Site> ContinuousConfiguration::positions_at(const double tau,
                                                        const Model &model) const {
  validate(model);
  if (!std::isfinite(tau) || tau < 0.0 || tau > model.beta) {
    throw std::invalid_argument("tau must lie in [0, beta]");
  }
  std::vector<Site> positions;
  positions.reserve(worldlines.size());
  for (const ContinuousPath &path : worldlines) {
    Site position = path.position_at(std::min(tau, path.duration));
    for (Coord &coordinate : position) {
      coordinate = torus_mod(coordinate, model.linear_size);
    }
    positions.push_back(std::move(position));
  }
  return positions;
}

Site ContinuousConfiguration::total_winding(const Model &model) const {
  validate(model);
  Site displacement(model.dimension, 0);
  for (const ContinuousPath &path : worldlines) {
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      const Coord path_displacement = detail::checked_subtract(
          path.end[axis], path.start[axis], "path displacement exceeds int64 range");
      displacement[axis] = detail::checked_add(displacement[axis], path_displacement,
                                               "total winding displacement exceeds int64 range");
    }
  }
  for (Coord &coordinate : displacement) {
    if (coordinate % model.linear_size != 0) {
      throw std::logic_error("total covering displacement is not an integer winding");
    }
    coordinate /= model.linear_size;
  }
  return displacement;
}

ContinuousConfiguration sample_ideal_continuous_configuration(const CanonicalEnsemble &ensemble,
                                                              Random &random,
                                                              const NumericalOptions &options) {
  const Model &model = ensemble.model();
  validate_continuous_model(model);
  options.validate();

  const std::vector<Cycle> cycles = ensemble.sample_cycles(random);
  std::vector<ParticleId> permutation(model.particle_count);
  std::vector<std::optional<ContinuousPath>> staged_paths(model.particle_count);

  for (const Cycle &cycle : cycles) {
    auto paths = detail::sample_paths_for_cycle(cycle, model, random, options);
    for (std::size_t index = 0; index < cycle.size(); ++index) {
      const ParticleId label = cycle[index];
      permutation[label] = cycle[(index + 1) % cycle.size()];
      staged_paths[label] = std::move(paths[index]);
    }
  }

  std::vector<ContinuousPath> worldlines;
  worldlines.reserve(model.particle_count);
  for (auto &path : staged_paths) {
    if (!path.has_value()) {
      throw std::logic_error("continuous cycle sampling left a particle path unassigned");
    }
    worldlines.push_back(std::move(*path));
  }

  ContinuousConfiguration configuration{
      .cycles = cycles,
      .permutation = std::move(permutation),
      .worldlines = std::move(worldlines),
      .log_Z0_N = ensemble.log_partition(model.particle_count),
  };
  configuration.validate(model);
  return configuration;
}

ContinuousConfiguration sample_ideal_continuous_configuration(const Model &model, Random &random,
                                                              const NumericalOptions &options) {
  return sample_ideal_continuous_configuration(CanonicalEnsemble(model), random, options);
}

} // namespace qmc
