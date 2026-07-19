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
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace qmc {

namespace detail {
class OccupancyIndex;
}

enum class MoveKind : std::uint8_t {
  SegmentMove = 0,
  CycleMove = 1,
  GlobalMove = 2,
  StitchMove = 3,
  TimeShiftMove = 4
};

[[nodiscard]] std::string_view move_name(MoveKind move) noexcept;

struct MoveStatistics {
  std::uint64_t attempts = 0;
  std::uint64_t accepts = 0;
  std::uint64_t topology_changes = 0;

  [[nodiscard]] std::optional<double> acceptance() const noexcept;
  [[nodiscard]] std::optional<double> topology_change_rate() const noexcept;
};

struct SweepOptions {
  // nullopt means one segment attempt per particle.
  std::optional<std::size_t> segment_updates;
  double segment_fraction = 0.25;
  std::size_t cycle_updates = 1;
  std::size_t global_updates = 0;
  std::size_t stitch_updates = 0;
  double stitch_fraction = 0.25;
  std::size_t stitch_locality_radius = 1;
  double stitch_global_partner_probability = 0.05;
  std::size_t time_shift_updates = 0;
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
  ~InteractingSampler();
  InteractingSampler(const InteractingSampler &other);
  InteractingSampler &operator=(const InteractingSampler &other);
  InteractingSampler(InteractingSampler &&) noexcept;
  InteractingSampler &operator=(InteractingSampler &&) noexcept;

  [[nodiscard]] bool
  segment_update(std::optional<ParticleId> particle = std::nullopt,
                 std::optional<std::pair<double, double>> interval = std::nullopt,
                 double fraction = 0.25);
  [[nodiscard]] bool whole_worldline_update(std::optional<ParticleId> particle = std::nullopt);
  [[nodiscard]] bool cycle_update(std::optional<std::size_t> cycle_index = std::nullopt);
  // Redraws two paths inside a slab from their exact ideal torus conditional.
  // An exchanged endpoint matching transposes their permutation successors.
  [[nodiscard]] bool stitch_update(std::optional<ParticleId> particle = std::nullopt,
                                   std::optional<ParticleId> partner = std::nullopt,
                                   std::optional<std::pair<double, double>> interval = std::nullopt,
                                   double fraction = 0.25, std::size_t locality_radius = 1,
                                   double global_partner_probability = 0.05);
  // Performs several random-scan stitch attempts at one seam, amortizing the
  // position bucket construction. nullopt means one attempt per particle.
  void stitch_sweep(std::optional<std::size_t> updates = std::nullopt, double fraction = 0.25,
                    std::optional<double> tau0 = std::nullopt, std::size_t locality_radius = 1,
                    double global_partner_probability = 0.05);
  // Rejection-free cyclic rotation of every closed permutation loop.
  [[nodiscard]] bool time_shift_update(std::optional<double> shift = std::nullopt);
  // Strictly reversible A B^m A macro-kernel: uniform time rotation, fixed-seam
  // stitch sweep, and a second independent uniform time rotation.
  void random_seam_stitch_sweep(std::optional<std::size_t> updates = std::nullopt,
                                double fraction = 0.35, std::size_t locality_radius = 1,
                                double global_partner_probability = 0.05);
  [[nodiscard]] bool global_update();
  void sweep(const SweepOptions &options = SweepOptions{});

  [[nodiscard]] InteractingObservables observables() const;
  [[nodiscard]] std::vector<InteractingObservables> run(std::size_t sample_count,
                                                        const RunOptions &options = RunOptions{});

  [[nodiscard]] const InteractingModel &model() const noexcept { return model_; }
  [[nodiscard]] const ContinuousConfiguration &state() const noexcept { return state_; }
  [[nodiscard]] double pair_overlap() const noexcept { return pair_overlap_; }
  [[nodiscard]] double action() const noexcept { return action_; }
  [[nodiscard]] const std::array<MoveStatistics, 5> &statistics() const noexcept {
    return statistics_;
  }
  [[nodiscard]] const MoveStatistics &statistics(MoveKind move) const;

private:
  using LabeledPath = std::pair<ParticleId, ContinuousPath>;

  struct StitchProposal {
    std::vector<LabeledPath> replacements;
    std::vector<ParticleId> permutation;
    bool exchanged = false;
  };

  bool try_path_replacements(std::vector<LabeledPath> replacements, MoveKind move);
  [[nodiscard]] StitchProposal sample_stitch_pair_proposal(ParticleId particle, ParticleId partner,
                                                           double tau0, double tau1);
  [[nodiscard]] bool try_stitch_pair(ParticleId particle, ParticleId partner, double tau0,
                                     double tau1);
  [[nodiscard]] bool metropolis_accept(double delta_action);
  void rebuild_occupancy_index();

  InteractingModel model_;
  NumericalOptions numerical_;
  Random random_;
  FreeBosonTable free_table_;
  ContinuousConfiguration state_;
  double pair_overlap_ = 0.0;
  double action_ = 0.0;
  std::array<MoveStatistics, 5> statistics_{};
  std::unique_ptr<detail::OccupancyIndex> occupancy_index_;
};

} // namespace qmc

#endif
