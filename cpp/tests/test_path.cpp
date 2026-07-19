#include "qmc/path.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <utility>
#include <vector>

namespace qmc {
namespace {

void expect_same_path(const ContinuousPath &actual, const ContinuousPath &expected) {
  EXPECT_DOUBLE_EQ(actual.duration, expected.duration);
  EXPECT_EQ(actual.start, expected.start);
  EXPECT_EQ(actual.end, expected.end);
  ASSERT_EQ(actual.events.size(), expected.events.size());
  for (std::size_t index = 0; index < actual.events.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_DOUBLE_EQ(actual.events[index].time, expected.events[index].time);
    EXPECT_EQ(actual.events[index].axis, expected.events[index].axis);
    EXPECT_EQ(actual.events[index].direction, expected.events[index].direction);
  }
}

std::vector<ContinuousPath>
reference_split_continuous_path(const ContinuousPath &path,
                                const std::span<const double> cut_times) {
  std::vector<double> boundaries;
  boundaries.reserve(cut_times.size() + 2);
  boundaries.push_back(0.0);
  boundaries.insert(boundaries.end(), cut_times.begin(), cut_times.end());
  boundaries.push_back(path.duration);

  std::vector<ContinuousPath> pieces;
  pieces.reserve(boundaries.size() - 1);
  for (std::size_t piece_index = 0; piece_index + 1 < boundaries.size(); ++piece_index) {
    const double left = boundaries[piece_index];
    const double right = boundaries[piece_index + 1];
    std::vector<JumpEvent> local_events;
    for (const JumpEvent &event : path.events) {
      if (event.time > left && event.time <= right) {
        local_events.push_back(JumpEvent{
            .time = event.time - left,
            .axis = event.axis,
            .direction = event.direction,
        });
      }
    }
    pieces.push_back(ContinuousPath{
        .duration = right - left,
        .start = path.position_at(left),
        .end = path.position_at(right),
        .events = std::move(local_events),
    });
  }
  return pieces;
}

ContinuousPath reference_resample_path_interval(const ContinuousPath &path, const double tau0,
                                                const double tau1, const double hopping,
                                                Random &random) {
  const ContinuousPath proposal = sample_continuous_bridge(
      path.position_at(tau0), path.position_at(tau1), tau1 - tau0, hopping, random);
  std::vector<JumpEvent> events;
  events.reserve(path.events.size() + proposal.events.size());
  for (const JumpEvent &event : path.events) {
    if (event.time <= tau0) {
      events.push_back(event);
    }
  }
  for (const JumpEvent &event : proposal.events) {
    events.push_back(JumpEvent{
        .time = tau0 + event.time,
        .axis = event.axis,
        .direction = event.direction,
    });
  }
  for (const JumpEvent &event : path.events) {
    if (event.time > tau1) {
      events.push_back(event);
    }
  }
  return ContinuousPath{
      .duration = path.duration,
      .start = path.start,
      .end = path.end,
      .events = std::move(events),
  };
}

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

TEST(ContinuousPathTest, SplitPreservesLegacyBoundarySemanticsWithCoincidentEvents) {
  const ContinuousPath path{
      .duration = 1.0,
      .start = {0},
      .end = {0},
      .events = {{.time = 0.0, .axis = 0, .direction = 1},
                 {.time = 0.0, .axis = 0, .direction = 1},
                 {.time = 0.5, .axis = 0, .direction = -1},
                 {.time = 0.5, .axis = 0, .direction = 1},
                 {.time = 1.0, .axis = 0, .direction = -1},
                 {.time = 1.0, .axis = 0, .direction = -1}},
  };
  path.validate(1);

  const std::vector<double> cuts{0.5};
  const auto pieces = split_continuous_path(path, cuts);
  ASSERT_EQ(pieces.size(), 2U);
  EXPECT_EQ(pieces[0].start, Site({2}));
  EXPECT_EQ(pieces[0].end, Site({2}));
  ASSERT_EQ(pieces[0].events.size(), 2U);
  EXPECT_DOUBLE_EQ(pieces[0].events[0].time, 0.5);
  EXPECT_DOUBLE_EQ(pieces[0].events[1].time, 0.5);
  EXPECT_EQ(pieces[1].start, Site({2}));
  EXPECT_EQ(pieces[1].end, Site({0}));
  ASSERT_EQ(pieces[1].events.size(), 2U);
  EXPECT_DOUBLE_EQ(pieces[1].events[0].time, 0.5);
  EXPECT_DOUBLE_EQ(pieces[1].events[1].time, 0.5);
}

TEST(ContinuousPathTest, SplitHandlesZeroDurationAfterApplyingTimeZeroEvents) {
  const ContinuousPath path{
      .duration = 0.0,
      .start = {3},
      .end = {4},
      .events = {{.time = 0.0, .axis = 0, .direction = 1}},
  };
  path.validate(1);

  const auto pieces = split_continuous_path(path, {});
  ASSERT_EQ(pieces.size(), 1U);
  EXPECT_DOUBLE_EQ(pieces[0].duration, 0.0);
  EXPECT_EQ(pieces[0].start, Site({4}));
  EXPECT_EQ(pieces[0].end, Site({4}));
  EXPECT_TRUE(pieces[0].events.empty());
}

TEST(ContinuousPathTest, CursorSplitMatchesPreviousTraversalExactly) {
  const ContinuousPath path{
      .duration = 1.0,
      .start = {0, 0},
      .end = {0, 1},
      .events = {{.time = 0.0, .axis = 0, .direction = 1},
                 {.time = 0.1, .axis = 1, .direction = -1},
                 {.time = 0.2, .axis = 0, .direction = 1},
                 {.time = 0.2, .axis = 1, .direction = 1},
                 {.time = 0.5, .axis = 0, .direction = -1},
                 {.time = 0.75, .axis = 1, .direction = 1},
                 {.time = 1.0, .axis = 0, .direction = -1}},
  };
  path.validate(2);
  const std::vector<double> cuts{0.2, 0.6, 0.75};

  const auto actual = split_continuous_path(path, cuts);
  const auto expected = reference_split_continuous_path(path, cuts);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    SCOPED_TRACE(index);
    expect_same_path(actual[index], expected[index]);
  }
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

TEST(ContinuousPathTest, IntervalResamplingRetainsLeftCutAndReplacesRightCutEvents) {
  const ContinuousPath path{
      .duration = 1.0,
      .start = {0},
      .end = {0},
      .events = {{.time = 0.0, .axis = 0, .direction = 1},
                 {.time = 0.25, .axis = 0, .direction = 1},
                 {.time = 0.25, .axis = 0, .direction = -1},
                 {.time = 0.5, .axis = 0, .direction = 1},
                 {.time = 0.5, .axis = 0, .direction = -1},
                 {.time = 0.75, .axis = 0, .direction = 1},
                 {.time = 0.75, .axis = 0, .direction = -1},
                 {.time = 0.9, .axis = 0, .direction = -1}},
  };
  path.validate(1);
  Random random(19);

  const ContinuousPath proposal = resample_path_interval(path, 0.25, 0.75, 0.0, random);
  ASSERT_EQ(proposal.events.size(), 4U);
  EXPECT_DOUBLE_EQ(proposal.events[0].time, 0.0);
  EXPECT_DOUBLE_EQ(proposal.events[1].time, 0.25);
  EXPECT_DOUBLE_EQ(proposal.events[2].time, 0.25);
  EXPECT_DOUBLE_EQ(proposal.events[3].time, 0.9);
  EXPECT_EQ(proposal.position_at(0.25), path.position_at(0.25));
  EXPECT_EQ(proposal.position_at(0.75), path.position_at(0.75));
}

TEST(ContinuousPathTest, CursorResamplingMatchesPreviousTraversalAndRandomStreamExactly) {
  Random source_random(812);
  const ContinuousPath path = sample_continuous_bridge({0, -1}, {3, 2}, 1.4, 0.9, source_random);
  Random actual_random(913);
  Random expected_random(913);

  const ContinuousPath actual = resample_path_interval(path, 0.3, 1.1, 0.9, actual_random);
  const ContinuousPath expected =
      reference_resample_path_interval(path, 0.3, 1.1, 0.9, expected_random);
  expect_same_path(actual, expected);
  EXPECT_DOUBLE_EQ(actual_random.uniform_open(), expected_random.uniform_open());
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
