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

  [[nodiscard]] bool operator==(const JumpEvent &) const = default;
};

// A piecewise-constant covering-space path. Positions are right-continuous:
// an event at tau is included in position_at(tau).
class ContinuousPath {
public:
  // Validates the complete path and takes ownership of its endpoints and
  // events. Throws std::invalid_argument for malformed path data and an
  // overflow exception when applying an event exceeds the coordinate range.
  ContinuousPath(double duration, Site start, Site end, std::vector<JumpEvent> events);

  // Expensive diagnostic audit, including compatibility with dimension.
  void validate(std::size_t dimension) const;
  [[nodiscard]] double duration() const noexcept { return duration_; }
  [[nodiscard]] std::size_t dimension() const noexcept { return start_.size(); }
  // These references and the event span borrow from this path and remain valid
  // until the path is assigned a different value or destroyed.
  [[nodiscard]] const Site &start() const noexcept { return start_; }
  [[nodiscard]] const Site &end() const noexcept { return end_; }
  [[nodiscard]] std::span<const JumpEvent> events() const noexcept { return events_; }
  [[nodiscard]] Site position_at(double tau) const;
  [[nodiscard]] std::vector<Site> positions_after_events() const;
  [[nodiscard]] ContinuousPath translated(const Site &displacement) const;
  [[nodiscard]] std::size_t event_count() const noexcept { return events_.size(); }
  [[nodiscard]] bool operator==(const ContinuousPath &) const = default;

private:
  double duration_;
  Site start_;
  Site end_;
  std::vector<JumpEvent> events_;
};

// Exact conditioned continuous-time free bridge on the covering lattice Z^d.
[[nodiscard]] ContinuousPath sample_continuous_bridge(const Site &a, const Site &b, double duration,
                                                      const FreePathKernels &kernels,
                                                      Random &random);

// One-off convenience overload.
[[nodiscard]] ContinuousPath
sample_continuous_bridge(const Site &a, const Site &b, double duration, double hopping,
                         Random &random, const NumericalOptions &options = NumericalOptions{});

// Exact continuous-time free bridge from covering representative a to the
// physical torus site b modulo linear_size. The covering endpoint, including
// its winding sector, is sampled as part of the bridge.
[[nodiscard]] ContinuousPath sample_continuous_bridge_torus(const Site &a, const Site &b,
                                                            double duration,
                                                            const FreePathKernels &kernels,
                                                            Random &random);

// One-off convenience overload.
[[nodiscard]] ContinuousPath
sample_continuous_bridge_torus(const Site &a, const Site &b, double duration, Coord linear_size,
                               double hopping, Random &random,
                               const NumericalOptions &options = NumericalOptions{});

// Log of exp(-2*hopping*duration*dimension) times the factorized torus kernel.
// The common exponential scale cancels in endpoint-matching probabilities.
[[nodiscard]] double log_torus_kernel_scaled(const Site &a, const Site &b, double duration,
                                             const FreePathKernels &kernels);

// One-off convenience overload.
[[nodiscard]] double log_torus_kernel_scaled(const Site &a, const Site &b, double duration,
                                             Coord linear_size, double hopping,
                                             const NumericalOptions &options = NumericalOptions{});

// Splits at strictly increasing internal times. An event at a cut belongs to
// the piece on its left; the next piece starts at the post-event position.
[[nodiscard]] std::vector<ContinuousPath> split_continuous_path(const ContinuousPath &path,
                                                                std::span<const double> cut_times);

// Replaces (tau0, tau1] by an exact free bridge with its covering endpoints fixed.
[[nodiscard]] ContinuousPath resample_path_interval(const ContinuousPath &path, double tau0,
                                                    double tau1, const FreePathKernels &kernels,
                                                    Random &random);

// One-off convenience overload.
[[nodiscard]] ContinuousPath
resample_path_interval(const ContinuousPath &path, double tau0, double tau1, double hopping,
                       Random &random, const NumericalOptions &options = NumericalOptions{});

} // namespace qmc

#endif
