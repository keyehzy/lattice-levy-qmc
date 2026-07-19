#ifndef QMC_INTERACTING_SAMPLER_HPP
#define QMC_INTERACTING_SAMPLER_HPP

#include "qmc/continuous_configuration.hpp"
#include "qmc/free_boson.hpp"
#include "qmc/interacting_model.hpp"
#include "qmc/model.hpp"
#include "qmc/random.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace qmc {

enum class MoveKind : std::uint8_t { SegmentMove = 0, CycleMove = 1, GlobalMove = 2 };

[[nodiscard]] std::string_view move_name(MoveKind move) noexcept;

struct MoveStatistics {
  std::uint64_t attempts = 0;
  std::uint64_t accepts = 0;

  [[nodiscard]] std::optional<double> acceptance() const noexcept;
};

struct SweepOptions {
  // nullopt means one segment attempt per particle.
  std::optional<std::size_t> segment_updates;
  double segment_fraction = 0.25;
  std::size_t cycle_updates = 1;
  std::size_t global_updates = 0;
};

struct RunOptions {
  std::size_t burn_in = 0;
  std::size_t thin = 1;
  SweepOptions sweep;
};

struct InteractingObservables {
  double action = 0.0;
  double pair_overlap_time = 0.0;
  double double_occupancy_per_site = 0.0;
  double kinetic_energy = 0.0;
  double interaction_energy = 0.0;
  double total_energy = 0.0;
  std::size_t event_count = 0;
  Site winding;
  std::vector<std::size_t> cycle_lengths;
};

class InteractingSampler {
public:
  InteractingSampler(InteractingModel model, NumericalOptions numerical, std::uint64_t seed);
  explicit InteractingSampler(InteractingModel model, std::uint64_t seed)
      : InteractingSampler(model, NumericalOptions{}, seed) {}

  [[nodiscard]] bool
  segment_update(std::optional<ParticleId> particle = std::nullopt,
                 std::optional<std::pair<double, double>> interval = std::nullopt,
                 double fraction = 0.25);
  [[nodiscard]] bool whole_worldline_update(std::optional<ParticleId> particle = std::nullopt);
  [[nodiscard]] bool cycle_update(std::optional<std::size_t> cycle_index = std::nullopt);
  [[nodiscard]] bool global_update();
  void sweep(const SweepOptions &options = SweepOptions{});

  [[nodiscard]] InteractingObservables observables() const;
  [[nodiscard]] std::vector<InteractingObservables> run(std::size_t sample_count,
                                                        const RunOptions &options = RunOptions{});

  [[nodiscard]] const InteractingModel &model() const noexcept { return model_; }
  [[nodiscard]] const ContinuousConfiguration &state() const noexcept { return state_; }
  [[nodiscard]] double pair_overlap() const noexcept { return pair_overlap_; }
  [[nodiscard]] double action() const noexcept { return action_; }
  [[nodiscard]] const std::array<MoveStatistics, 3> &statistics() const noexcept {
    return statistics_;
  }
  [[nodiscard]] const MoveStatistics &statistics(MoveKind move) const;

private:
  using LabeledPath = std::pair<ParticleId, ContinuousPath>;

  bool try_path_replacements(std::vector<LabeledPath> replacements, MoveKind move);
  [[nodiscard]] bool metropolis_accept(double delta_action);

  InteractingModel model_;
  NumericalOptions numerical_;
  Random random_;
  FreeBosonTable free_table_;
  ContinuousConfiguration state_;
  double pair_overlap_ = 0.0;
  double action_ = 0.0;
  std::array<MoveStatistics, 3> statistics_{};
};

} // namespace qmc

#endif
