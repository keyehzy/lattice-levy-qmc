#ifndef QMC_PATH_HPP
#define QMC_PATH_HPP

#include "qmc/free_numerics.hpp"
#include "qmc/model.hpp"
#include "qmc/random.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qmc {

using Axis = std::uint32_t;

struct JumpEvent {
  double time = 0.0;
  Axis axis = 0;
  std::int8_t direction = 1;
};

// A piecewise-constant covering-space path. Positions are right-continuous:
// an event at tau is included in position_at(tau).
struct ContinuousPath {
  double duration = 0.0;
  Site start;
  Site end;
  std::vector<JumpEvent> events;

  void validate(std::size_t dimension) const;
  [[nodiscard]] Site position_at(double tau) const;
  [[nodiscard]] std::vector<Site> positions_after_events() const;
  [[nodiscard]] ContinuousPath translated(const Site &displacement) const;
  [[nodiscard]] std::size_t event_count() const noexcept { return events.size(); }
};

// Exact conditioned continuous-time free bridge on the covering lattice Z^d.
[[nodiscard]] ContinuousPath
sample_continuous_bridge(const Site &a, const Site &b, double duration, double hopping,
                         Random &random, const NumericalOptions &options = NumericalOptions{});

// Splits at strictly increasing internal times. An event at a cut belongs to
// the piece on its left; the next piece starts at the post-event position.
[[nodiscard]] std::vector<ContinuousPath> split_continuous_path(const ContinuousPath &path,
                                                                std::span<const double> cut_times);

// Replaces (tau0, tau1] by an exact free bridge with its covering endpoints fixed.
[[nodiscard]] ContinuousPath
resample_path_interval(const ContinuousPath &path, double tau0, double tau1, double hopping,
                       Random &random, const NumericalOptions &options = NumericalOptions{});

} // namespace qmc

#endif
