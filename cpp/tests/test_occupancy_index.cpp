#include "occupancy_index.hpp"
#include "qmc/model.hpp"
#include "qmc/path.hpp"

#include <array>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace qmc::detail {
namespace {

ContinuousPath stationary_path(const Coord site, const double duration = 1.0) {
  return ContinuousPath(duration, {site}, {site}, {});
}

Model occupancy_model() {
  return Model{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  };
}

TEST(OccupancyIndexTest, StagesAffectedTimelinesAndPublishesOnlyOnCommit) {
  const Model model = occupancy_model();
  std::vector<ContinuousPath> accepted{stationary_path(0), stationary_path(0)};
  std::vector<ContinuousPath> proposed = accepted;
  proposed[0] = stationary_path(1);

  OccupancyIndex index(model);
  index.rebuild(accepted);
  ASSERT_TRUE(index.represents(accepted));
  ASSERT_DOUBLE_EQ(index.pair_overlap(), 1.0);

  const std::array replacements{
      PathReplacementView{.label = 0, .old_path = accepted[0], .new_path = proposed[0]},
  };
  auto transaction = index.begin_replacement(replacements, 1.0);
  EXPECT_DOUBLE_EQ(transaction.proposed_overlap(), 0.0);
  EXPECT_DOUBLE_EQ(transaction.exact_proposed_overlap(), 0.0);
  EXPECT_TRUE(index.represents(accepted));
  EXPECT_FALSE(index.represents(proposed));
  EXPECT_DOUBLE_EQ(index.pair_overlap(), 1.0);

  transaction.commit();
  EXPECT_THROW(static_cast<void>(transaction.exact_proposed_overlap()), std::logic_error);
  EXPECT_FALSE(index.represents(accepted));
  EXPECT_TRUE(index.represents(proposed));
  EXPECT_DOUBLE_EQ(index.pair_overlap(), 0.0);
}

TEST(OccupancyIndexTest, InjectedFailuresAfterStagingLeaveAcceptedLedgerUntouched) {
  const Model model = occupancy_model();
  const std::vector<ContinuousPath> accepted{stationary_path(0), stationary_path(0)};
  const ContinuousPath proposed = stationary_path(1);
  const std::array replacements{
      PathReplacementView{.label = 0, .old_path = accepted[0], .new_path = proposed},
  };

  OccupancyIndex index(model);
  index.rebuild(accepted);
  for (const bool topology_failure : {false, true}) {
    try {
      auto transaction = index.begin_replacement(replacements, 1.0);
      ASSERT_DOUBLE_EQ(transaction.proposed_overlap(), 0.0);
      if (topology_failure) {
        throw std::invalid_argument("injected topology preparation failure");
      }
      throw std::overflow_error("injected action calculation failure");
    } catch (const std::exception &) {
      // An abandoned transaction is the rollback: accepted storage was never
      // touched and there is no potentially throwing undo path.
    }
    EXPECT_TRUE(index.represents(accepted));
    EXPECT_DOUBLE_EQ(index.pair_overlap(), 1.0);
  }
}

TEST(OccupancyIndexTest, MidReplacementFailureCannotPartiallyRemoveAcceptedPaths) {
  const Model model = occupancy_model();
  const std::vector<ContinuousPath> accepted{stationary_path(0), stationary_path(0)};
  const ContinuousPath proposed = stationary_path(1);
  const ContinuousPath unoccupied_old_path = stationary_path(2);
  const std::array replacements{
      PathReplacementView{.label = 0, .old_path = accepted[0], .new_path = proposed},
      PathReplacementView{.label = 1, .old_path = unoccupied_old_path, .new_path = accepted[1]},
  };

  OccupancyIndex index(model);
  index.rebuild(accepted);
  EXPECT_THROW(static_cast<void>(index.begin_replacement(replacements, 1.0)), std::logic_error);
  EXPECT_TRUE(index.represents(accepted));
  EXPECT_DOUBLE_EQ(index.pair_overlap(), 1.0);
}

} // namespace
} // namespace qmc::detail
