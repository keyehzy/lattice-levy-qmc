#ifndef QMC_SRC_OCCUPANCY_INDEX_HPP
#define QMC_SRC_OCCUPANCY_INDEX_HPP

#include "qmc/model.hpp"
#include "qmc/path.hpp"

#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace qmc::detail {

// Sparse space-time occupancy ledger used to evaluate exact local changes in
// pair-overlap action when a small set of paths is replaced.
class OccupancyIndex {
public:
  explicit OccupancyIndex(const Model &model);

  void rebuild(std::span<const ContinuousPath> paths);
  [[nodiscard]] double replace_paths(std::span<const ContinuousPath *const> old_paths,
                                     std::span<const ContinuousPath *const> new_paths,
                                     double current_overlap);
  void rollback_replacement(std::span<const ContinuousPath *const> old_paths,
                            std::span<const ContinuousPath *const> new_paths);
  [[nodiscard]] double pair_overlap();

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
  };

  [[nodiscard]] Site site_key(const Site &position) const;
  [[nodiscard]] SiteTimeline &timeline(const Site &key);
  void adjust_path(const ContinuousPath &path, std::int64_t sign);
  [[nodiscard]] double integrate_path_occupancy(const ContinuousPath &path);

  Coord linear_size_;
  std::size_t dimension_;
  double beta_;
  std::map<Site, SiteTimeline> timelines_;
};

} // namespace qmc::detail

#endif
