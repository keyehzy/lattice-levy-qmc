#ifndef QMC_SRC_OCCUPANCY_INDEX_HPP
#define QMC_SRC_OCCUPANCY_INDEX_HPP

#include "qmc/model.hpp"
#include "qmc/path.hpp"
#include "qmc/torus_layout.hpp"

#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace qmc::detail {

// One labeled accepted/proposed path pair. Keeping the pair in one value makes
// it impossible for occupancy updates to misalign parallel old/new spans.
struct PathReplacementView {
  ParticleId label;
  const ContinuousPath &old_path;
  const ContinuousPath &new_path;
};

// Sparse space-time occupancy ledger used to evaluate exact local changes in
// pair-overlap action when a small set of paths is replaced.
class OccupancyIndex {
private:
  struct SiteTimeline {
    std::int64_t initial = 0;
    std::map<double, std::int64_t> deltas;
    bool dirty = true;
    std::vector<double> times;
    std::vector<double> areas_before;
    std::vector<std::int64_t> occupancies_after;

    void adjust_initial(std::int64_t delta);
    void adjust_event(double event_time, std::int64_t delta);
    void rebuild();
    [[nodiscard]] double integral_to(double tau);
    [[nodiscard]] double integral(double tau0, double tau1);
    [[nodiscard]] double pair_integral(double beta);
    [[nodiscard]] bool empty() const noexcept;
  };

  using TimelineMap = std::map<SiteId, SiteTimeline>;

public:
  class ReplacementTransaction {
  public:
    ReplacementTransaction(const ReplacementTransaction &) = delete;
    ReplacementTransaction &operator=(const ReplacementTransaction &) = delete;
    ReplacementTransaction(ReplacementTransaction &&other) noexcept;
    ReplacementTransaction &operator=(ReplacementTransaction &&) = delete;
    ~ReplacementTransaction() = default;

    [[nodiscard]] double proposed_overlap() const noexcept { return proposed_overlap_; }

    // Publishes prepared timeline nodes without allocation. Destroying an
    // uncommitted transaction leaves the accepted index byte-for-byte intact.
    void commit() noexcept;

  private:
    friend class OccupancyIndex;

    ReplacementTransaction(OccupancyIndex &owner, double proposed_overlap,
                           TimelineMap staged_timelines) noexcept;

    OccupancyIndex *owner_;
    double proposed_overlap_;
    TimelineMap staged_timelines_;
  };

  explicit OccupancyIndex(const Model &model);

  void rebuild(std::span<const ContinuousPath> paths);
  [[nodiscard]] ReplacementTransaction
  begin_replacement(std::span<const PathReplacementView> replacements, double current_overlap);
  [[nodiscard]] double pair_overlap();

  // Expensive structural audit used at fault and rejection boundaries.
  [[nodiscard]] bool represents(std::span<const ContinuousPath> paths) const;

private:
  [[nodiscard]] static SiteTimeline &timeline(TimelineMap &timelines, SiteId key);
  void stage_path_timelines(const ContinuousPath &path, TimelineMap &staged) const;
  void adjust_path(TimelineMap &timelines, const ContinuousPath &path, std::int64_t sign) const;
  [[nodiscard]] double integrate_path_occupancy(TimelineMap &timelines,
                                                const ContinuousPath &path) const;
  [[nodiscard]] static bool same_occupancies(const TimelineMap &left,
                                             const TimelineMap &right) noexcept;

  TorusLayout layout_;
  double beta_;
  TimelineMap timelines_;
};

} // namespace qmc::detail

#endif
