#ifndef QMC_INTERACTING_SAMPLER_HPP
#define QMC_INTERACTING_SAMPLER_HPP

#include "qmc/continuous_configuration.hpp"
#include "qmc/free_boson.hpp"
#include "qmc/interacting_model.hpp"
#include "qmc/model.hpp"
#include "qmc/random.hpp"
#include "qmc/torus_layout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace qmc {

namespace detail {
class AcceptedChainState;
struct InteractingSamplerTestAccess;
} // namespace detail

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
  std::uint64_t successor_changes = 0;

  [[nodiscard]] std::optional<double> acceptance() const noexcept;
  [[nodiscard]] std::optional<double> topology_change_rate() const noexcept;
  [[nodiscard]] std::optional<double> successor_changes_per_attempt() const noexcept;
};

struct StitchMixture {
  // Each count must be unique and in [2, 8]. Counts larger than N are ignored.
  std::vector<std::size_t> strand_counts{2};
  // Empty selects equal weights. Otherwise this must match strand_counts.
  std::vector<double> strand_weights;
};

struct SegmentUpdateOptions {
  // interval, when present, is an explicit [tau0, tau1] inside [0, beta].
  // Otherwise fraction selects the interval duration and must lie in (0, 1].
  // Every field is validated before random-number consumption, including
  // fraction when interval is present.
  std::optional<ParticleId> particle;
  std::optional<std::pair<double, double>> interval;
  double fraction = 0.25;
};

struct StitchUpdateOptions {
  // Empty strands selects a reversible spatially local set. Otherwise strands
  // must contain exactly strand_count distinct labels and anchor must be empty.
  // anchor optionally fixes the first label of an automatically selected set.
  // Every field is validated before random-number consumption.
  std::size_t strand_count = 2;
  std::optional<ParticleId> anchor;
  std::vector<ParticleId> strands;
  std::optional<std::pair<double, double>> interval;
  double fraction = 0.25;
  std::size_t locality_radius = 1;
  double global_partner_probability = 0.05;
};

struct StitchSweepOptions {
  // nullopt means one stitch attempt per particle.
  // tau0 fixes the shared left seam; otherwise it is sampled uniformly.
  std::optional<std::size_t> updates;
  double fraction = 0.25;
  std::optional<double> tau0;
  std::size_t locality_radius = 1;
  double global_partner_probability = 0.05;
  StitchMixture mixture;
};

struct RandomSeamStitchOptions {
  // nullopt means one stitch attempt per particle.
  // Every field is validated before the first time-origin rotation.
  std::optional<std::size_t> updates;
  double fraction = 0.35;
  std::size_t locality_radius = 1;
  double global_partner_probability = 0.05;
  StitchMixture mixture;
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
  StitchMixture stitch_mixture;
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

  [[nodiscard]] bool segment_update(const SegmentUpdateOptions &options = SegmentUpdateOptions{});
  [[nodiscard]] bool whole_worldline_update(std::optional<ParticleId> particle = std::nullopt);
  [[nodiscard]] bool cycle_update(std::optional<std::size_t> cycle_index = std::nullopt);
  // Redraws 2 <= k <= 8 paths inside a slab from the exact
  // permanent-normalized ideal torus conditional. A valid pair request is a
  // no-op when the model contains fewer than two particles.
  [[nodiscard]] bool stitch_update(const StitchUpdateOptions &options = StitchUpdateOptions{});
  // Performs several random-scan stitch attempts at one seam, amortizing the
  // position bucket construction.
  void stitch_sweep(const StitchSweepOptions &options = StitchSweepOptions{});
  // Rejection-free cyclic rotation of every closed permutation loop.
  [[nodiscard]] bool time_shift_update(std::optional<double> shift = std::nullopt);
  // Strictly reversible A B^m A macro-kernel: uniform time rotation, fixed-seam
  // stitch sweep, and a second independent uniform time rotation.
  void random_seam_stitch_sweep(const RandomSeamStitchOptions &options = RandomSeamStitchOptions{});
  [[nodiscard]] bool global_update();
  void sweep(const SweepOptions &options = SweepOptions{});

  [[nodiscard]] InteractingObservables observables() const;
  [[nodiscard]] std::vector<InteractingObservables> run(std::size_t sample_count,
                                                        const RunOptions &options = RunOptions{});

  [[nodiscard]] const InteractingModel &model() const noexcept { return model_; }
  [[nodiscard]] const ContinuousConfiguration &state() const noexcept;
  [[nodiscard]] double pair_overlap() const noexcept;
  [[nodiscard]] double action() const noexcept;
  [[nodiscard]] const std::array<MoveStatistics, 5> &statistics() const noexcept {
    return statistics_;
  }
  [[nodiscard]] const MoveStatistics &statistics(MoveKind move) const;

private:
  friend struct detail::InteractingSamplerTestAccess;

  using LabeledPath = std::pair<ParticleId, ContinuousPath>;

  struct StitchProposal {
    std::vector<LabeledPath> replacements;
    std::vector<ParticleId> successors;
    std::size_t successor_changes = 0;
  };

  struct LocalProposal {
    std::vector<LabeledPath> replacements;
    std::optional<std::vector<ParticleId>> successors;
    std::size_t successor_changes = 0;
  };

  struct PreparedStitchMixture {
    std::vector<std::size_t> counts;
    std::vector<double> weights;
  };

  struct PreparedStitchSweep {
    std::size_t updates = 0;
    double duration = 0.0;
    std::optional<double> tau0;
    std::size_t locality_radius = 0;
    double global_partner_probability = 0.0;
    PreparedStitchMixture mixture;
  };

  struct PreparedSweep {
    std::size_t segment_updates = 0;
    double segment_fraction = 0.0;
    std::size_t cycle_updates = 0;
    std::size_t global_updates = 0;
    PreparedStitchSweep stitch;
    std::size_t time_shift_updates = 0;
  };

  bool try_path_replacements(std::vector<LabeledPath> replacements, MoveKind move);
  bool try_proposal(LocalProposal proposal, MoveStatistics &move_statistics);
  [[nodiscard]] StitchProposal sample_stitch_proposal(std::span<const ParticleId> strands,
                                                      double tau0, double tau1);
  [[nodiscard]] bool try_stitch_strands(std::span<const ParticleId> strands, double tau0,
                                        double tau1);
  [[nodiscard]] bool metropolis_accept(double delta_action);
  void validate_segment_update_options(const SegmentUpdateOptions &options) const;
  void validate_stitch_update_options(const StitchUpdateOptions &options) const;
  [[nodiscard]] PreparedStitchMixture prepare_stitch_mixture(const StitchMixture &mixture) const;
  [[nodiscard]] PreparedStitchSweep prepare_stitch_sweep(const StitchSweepOptions &options) const;
  [[nodiscard]] PreparedSweep prepare_sweep(const SweepOptions &options) const;
  [[nodiscard]] bool execute_segment_update(const SegmentUpdateOptions &options);
  [[nodiscard]] bool execute_stitch_update(const StitchUpdateOptions &options);
  void execute_stitch_sweep(const PreparedStitchSweep &options);
  void execute_sweep(const PreparedSweep &options);

  InteractingModel model_;
  TorusLayout layout_;
  NumericalOptions numerical_;
  Random random_;
  CanonicalEnsemble free_ensemble_;
  std::unique_ptr<detail::AcceptedChainState> accepted_state_;
  std::array<MoveStatistics, 5> statistics_{};
};

} // namespace qmc

#endif
