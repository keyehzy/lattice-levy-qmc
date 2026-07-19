#include "qmc/path.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace qmc {
namespace {

TEST(ContinuousPathTest, UsesRightContinuousEventSemantics) {
  const ContinuousPath path{
      .duration = 1.0,
      .start = {0},
      .end = {0},
      .events = {{.time = 0.25, .axis = 0, .direction = 1},
                 {.time = 0.75, .axis = 0, .direction = -1}},
  };
  path.validate(1);
  EXPECT_EQ(path.position_at(0.0), Site({0}));
  EXPECT_EQ(path.position_at(0.249), Site({0}));
  EXPECT_EQ(path.position_at(0.25), Site({1}));
  EXPECT_EQ(path.position_at(0.75), Site({0}));
  EXPECT_EQ(path.position_at(1.0), Site({0}));
  EXPECT_EQ(path.positions_after_events(), std::vector<Site>({{1}, {0}}));

  const auto translated = path.translated({-3});
  EXPECT_EQ(translated.start, Site({-3}));
  EXPECT_EQ(translated.end, Site({-3}));
  EXPECT_EQ(translated.position_at(0.5), Site({-2}));
}

TEST(ContinuousPathTest, RejectsMalformedEventsAndEndpoints) {
  ContinuousPath path{
      .duration = 1.0,
      .start = {0},
      .end = {1},
      .events = {{.time = 0.5, .axis = 0, .direction = 1}},
  };
  EXPECT_NO_THROW(path.validate(1));
  path.events[0].direction = 2;
  EXPECT_THROW(path.validate(1), std::invalid_argument);
  path.events[0].direction = 1;
  path.events[0].axis = 1;
  EXPECT_THROW(path.validate(1), std::invalid_argument);
  path.events[0].axis = 0;
  path.events[0].time = 1.1;
  EXPECT_THROW(path.validate(1), std::invalid_argument);
  path.events[0].time = 0.5;
  path.end = {2};
  EXPECT_THROW(path.validate(1), std::invalid_argument);
}

TEST(ContinuousPathTest, SamplesStrictlyInternalEventsAndRequestedEndpoint) {
  Random random(7);
  for (int sample = 0; sample < 100; ++sample) {
    const auto path = sample_continuous_bridge({0, -2}, {5, 3}, 1.4, 0.8, random);
    path.validate(2);
    EXPECT_EQ(path.start, Site({0, -2}));
    EXPECT_EQ(path.end, Site({5, 3}));
    EXPECT_EQ(path.position_at(path.duration), path.end);
    EXPECT_TRUE(std::ranges::all_of(path.events, [&path](const JumpEvent &event) {
      return event.time > 0.0 && event.time < path.duration;
    }));
  }
}

TEST(ContinuousPathTest, TorusBridgeSamplesCoveringEndpointAtRequestedPhysicalSite) {
  Random random(71);
  for (int sample = 0; sample < 100; ++sample) {
    const auto path = sample_continuous_bridge_torus({7, -3}, {1, 4}, 0.7, 6, 1.1, random);
    path.validate(2);
    EXPECT_EQ(Site({torus_mod(path.start[0], 6), torus_mod(path.start[1], 6)}), Site({1, 3}));
    EXPECT_EQ(Site({torus_mod(path.end[0], 6), torus_mod(path.end[1], 6)}), Site({1, 4}));
    EXPECT_EQ(path.position_at(path.duration), path.end);
  }
}

TEST(ContinuousPathTest, SplitsCutEventsIntoTheLeftPiece) {
  const ContinuousPath path{
      .duration = 1.0,
      .start = {0},
      .end = {0},
      .events = {{.time = 0.5, .axis = 0, .direction = 1},
                 {.time = 0.75, .axis = 0, .direction = -1}},
  };
  const std::vector<double> cuts{0.5};
  const auto pieces = split_continuous_path(path, cuts);
  ASSERT_EQ(pieces.size(), 2U);
  ASSERT_EQ(pieces[0].events.size(), 1U);
  EXPECT_DOUBLE_EQ(pieces[0].events[0].time, 0.5);
  EXPECT_EQ(pieces[0].start, Site({0}));
  EXPECT_EQ(pieces[0].end, Site({1}));
  EXPECT_EQ(pieces[1].start, Site({1}));
  EXPECT_EQ(pieces[1].end, Site({0}));
  EXPECT_EQ(pieces[0].event_count() + pieces[1].event_count(), path.event_count());
}

TEST(ContinuousPathTest, IntervalResamplingPreservesOuterPathAndEndpoints) {
  Random random(10);
  const auto path = sample_continuous_bridge({0, 0}, {2, -1}, 2.0, 0.9, random);
  constexpr double tau0 = 0.4;
  constexpr double tau1 = 1.6;
  const Site left = path.position_at(tau0);
  const Site right = path.position_at(tau1);
  const auto proposal = resample_path_interval(path, tau0, tau1, 0.9, random);
  EXPECT_EQ(proposal.start, path.start);
  EXPECT_EQ(proposal.end, path.end);
  EXPECT_EQ(proposal.position_at(tau0), left);
  EXPECT_EQ(proposal.position_at(tau1), right);

  std::vector<JumpEvent> old_outer;
  std::vector<JumpEvent> new_outer;
  for (const JumpEvent &event : path.events) {
    if (event.time <= tau0 || event.time > tau1) {
      old_outer.push_back(event);
    }
  }
  for (const JumpEvent &event : proposal.events) {
    if (event.time <= tau0 || event.time > tau1) {
      new_outer.push_back(event);
    }
  }
  ASSERT_EQ(new_outer.size(), old_outer.size());
  for (std::size_t index = 0; index < old_outer.size(); ++index) {
    EXPECT_DOUBLE_EQ(new_outer[index].time, old_outer[index].time);
    EXPECT_EQ(new_outer[index].axis, old_outer[index].axis);
    EXPECT_EQ(new_outer[index].direction, old_outer[index].direction);
  }
}

TEST(ContinuousPathTest, HandlesZeroWeightAndRejectsBadIntervals) {
  Random random(1);
  const auto constant = sample_continuous_bridge({2}, {2}, 0.0, 1.0, random);
  EXPECT_TRUE(constant.events.empty());
  EXPECT_THROW(static_cast<void>(sample_continuous_bridge({2}, {3}, 0.0, 1.0, random)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(sample_continuous_bridge({2}, {3}, 1.0, 0.0, random)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(resample_path_interval(constant, 0.0, 0.0, 1.0, random)),
               std::invalid_argument);
}

} // namespace
} // namespace qmc
