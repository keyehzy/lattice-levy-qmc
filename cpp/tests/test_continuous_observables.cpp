#include "continuous_test_fixtures.hpp"
#include "qmc/continuous_observables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace qmc {
namespace {

static_assert(!std::default_initializable<ContinuousMeasurementContext>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousMeasurementContext &>().model()),
                             const Model &>);
static_assert(
    std::is_same_v<decltype(std::declval<const ContinuousMeasurementContext &>().layout()),
                   const TorusLayout &>);
static_assert(
    std::is_same_v<decltype(std::declval<const ContinuousMeasurementContext &>().seam_positions()),
                   std::span<const SiteId>>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousMeasurementContext &>().hops()),
                             std::span<const ContinuousHop>>);

ContinuousMeasurementContext make_owned_context() {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  return ContinuousMeasurementContext(configuration);
}

TEST(ContinuousMeasurementContextTest, OwnsEmptyConfigurationProvenance) {
  const Model model(ModelParameters{
      .particle_count = 0,
      .beta = 1.25,
      .linear_size = 4,
      .dimension = 2,
      .hopping = 0.75,
  });
  const ContinuousConfiguration configuration(model, Permutation(), {});
  const ContinuousMeasurementContext context(configuration);

  EXPECT_EQ(context.model(), model);
  EXPECT_EQ(context.layout(), TorusLayout(4, 2));
  EXPECT_TRUE(context.seam_positions().empty());
  EXPECT_TRUE(context.hops().empty());
  EXPECT_EQ(context.event_group_count(), 0U);
  EXPECT_THROW(static_cast<void>(context.event_time(0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(context.hops_at(0)), std::out_of_range);
}

TEST(ContinuousMeasurementContextTest, OwnsGeometryAfterConfigurationLifetimeEnds) {
  const ContinuousMeasurementContext context = make_owned_context();

  EXPECT_EQ(context.model().particle_count(), 2U);
  EXPECT_TRUE(std::ranges::equal(context.seam_positions(), std::array{SiteId(0), SiteId(1)}));
  EXPECT_EQ(context.hops().size(), 12U);
  EXPECT_EQ(context.event_group_count(), 5U);
}

TEST(ContinuousMeasurementContextTest, GroupsStableHopGeometryAtAllSeamSides) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const ContinuousMeasurementContext context(configuration);
  const std::array expected_times{0.0, 0.25, 0.5, 0.75, 1.0};
  const std::array<std::size_t, 5> expected_sizes{2, 3, 1, 3, 3};

  ASSERT_EQ(context.hops().size(), configuration.event_count());
  ASSERT_EQ(context.event_group_count(), expected_times.size());
  for (std::size_t group = 0; group < expected_times.size(); ++group) {
    EXPECT_DOUBLE_EQ(context.event_time(group), expected_times[group]);
    const auto hops = context.hops_at(group);
    ASSERT_EQ(hops.size(), expected_sizes[group]);
    EXPECT_TRUE(
        std::ranges::all_of(hops, [expected = expected_times[group]](const ContinuousHop &hop) {
          return hop.time == expected;
        }));
  }

  const auto zero = context.hops_at(0);
  EXPECT_EQ(zero[0], (ContinuousHop{.time = 0.0,
                                    .particle = 0,
                                    .departure = SiteId(0),
                                    .arrival = SiteId(1),
                                    .axis = 0,
                                    .direction = 1}));
  EXPECT_EQ(zero[1], (ContinuousHop{.time = 0.0,
                                    .particle = 1,
                                    .departure = SiteId(1),
                                    .arrival = SiteId(0),
                                    .axis = 0,
                                    .direction = -1}));
  const auto quarter = context.hops_at(1);
  ASSERT_EQ(quarter.size(), 3U);
  EXPECT_EQ(quarter[0].particle, 0U);
  EXPECT_EQ(quarter[0].direction, 1);
  EXPECT_EQ(quarter[1].particle, 0U);
  EXPECT_EQ(quarter[1].direction, -1);
  EXPECT_EQ(quarter[2].particle, 1U);
  EXPECT_EQ(quarter[2].direction, 1);

  EXPECT_THROW(static_cast<void>(context.event_time(expected_times.size())), std::out_of_range);
  EXPECT_THROW(static_cast<void>(context.hops_at(expected_times.size())), std::out_of_range);
}

TEST(ContinuousMeasurementContextTest, SeamPositionsPrecedeZeroEventsAndReplayClosesTopology) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const ContinuousMeasurementContext context(configuration);
  std::vector<SiteId> replay(context.seam_positions().begin(), context.seam_positions().end());

  const auto zero_hops = context.hops_at(0);
  for (const ContinuousHop &hop : zero_hops) {
    ASSERT_EQ(replay[hop.particle], hop.departure);
    replay[hop.particle] = hop.arrival;
  }
  const auto positions_at_zero = configuration.positions_at(0.0);
  ASSERT_EQ(positions_at_zero.size(), replay.size());
  for (std::size_t particle = 0; particle < replay.size(); ++particle) {
    EXPECT_EQ(replay[particle], context.layout().encode_covering(positions_at_zero[particle]));
  }

  for (std::size_t group = 1; group < context.event_group_count(); ++group) {
    for (const ContinuousHop &hop : context.hops_at(group)) {
      ASSERT_EQ(replay[hop.particle], hop.departure);
      EXPECT_EQ(hop.arrival, context.layout().shifted(hop.departure, hop.axis,
                                                      static_cast<Coord>(hop.direction)));
      replay[hop.particle] = hop.arrival;
    }
  }
  for (std::size_t particle = 0; particle < replay.size(); ++particle) {
    const ParticleId successor =
        configuration.topology().successor(static_cast<ParticleId>(particle));
    EXPECT_EQ(replay[particle], context.seam_positions()[successor]);
  }
}

TEST(ContinuousMeasurementContextTest, NormalizesOnlyAllowedOvershootToModelBeta) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const double duration_above = std::nextafter(model.beta(), 2.0);
  const ContinuousConfiguration above(
      model, Permutation({0}),
      {ContinuousPath(duration_above, {0}, {0},
                      {{.time = duration_above, .axis = 0, .direction = 1},
                       {.time = duration_above, .axis = 0, .direction = -1}})});
  const ContinuousMeasurementContext above_context(above);
  ASSERT_EQ(above_context.event_group_count(), 1U);
  EXPECT_DOUBLE_EQ(above_context.event_time(0), model.beta());

  const double duration_below = std::nextafter(model.beta(), 0.0);
  const ContinuousConfiguration below(
      model, Permutation({0}),
      {ContinuousPath(duration_below, {0}, {0},
                      {{.time = duration_below, .axis = 0, .direction = 1},
                       {.time = duration_below, .axis = 0, .direction = -1}})});
  const ContinuousMeasurementContext below_context(below);
  ASSERT_EQ(below_context.event_group_count(), 1U);
  EXPECT_DOUBLE_EQ(below_context.event_time(0), duration_below);
}

TEST(ContinuousMeasurementContextTest, RetainsDirectionWhenPeriodicArrivalsCoincide) {
  const Model model(ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousPath positive(
      1.0, {0}, {0},
      {{.time = 0.25, .axis = 0, .direction = 1}, {.time = 0.75, .axis = 0, .direction = -1}});
  const ContinuousPath negative(
      1.0, {0}, {0},
      {{.time = 0.25, .axis = 0, .direction = -1}, {.time = 0.75, .axis = 0, .direction = 1}});
  const ContinuousConfiguration configuration(model, Permutation({0, 1}), {positive, negative});
  const ContinuousMeasurementContext context(configuration);
  const auto first_group = context.hops_at(0);

  ASSERT_EQ(first_group.size(), 2U);
  EXPECT_EQ(first_group[0].departure, SiteId(0));
  EXPECT_EQ(first_group[1].departure, SiteId(0));
  EXPECT_EQ(first_group[0].arrival, SiteId(1));
  EXPECT_EQ(first_group[1].arrival, SiteId(1));
  EXPECT_EQ(first_group[0].direction, 1);
  EXPECT_EQ(first_group[1].direction, -1);
}

TEST(ContinuousMeasurementContextTest, HandlesSingleSiteAndMultidimensionalPhysicalGeometry) {
  const Model single_site_model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 1,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousConfiguration single_site(
      single_site_model, Permutation({0}),
      {ContinuousPath(
          1.0, {0}, {0},
          {{.time = 0.5, .axis = 0, .direction = 1}, {.time = 0.5, .axis = 0, .direction = -1}})});
  const ContinuousMeasurementContext single_site_context(single_site);
  const auto single_site_hops = single_site_context.hops_at(0);
  ASSERT_EQ(single_site_hops.size(), 2U);
  for (const ContinuousHop &hop : single_site_hops) {
    EXPECT_EQ(hop.departure, SiteId(0));
    EXPECT_EQ(hop.arrival, SiteId(0));
  }
  EXPECT_EQ(single_site_hops[0].direction, 1);
  EXPECT_EQ(single_site_hops[1].direction, -1);

  const Model plane_model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 2,
      .hopping = 1.0,
  });
  const ContinuousConfiguration plane(plane_model, Permutation({0}),
                                      {ContinuousPath(1.0, {-1, 4}, {-1, 4},
                                                      {{.time = 0.2, .axis = 1, .direction = 1},
                                                       {.time = 0.3, .axis = 0, .direction = -1},
                                                       {.time = 0.4, .axis = 1, .direction = -1},
                                                       {.time = 0.5, .axis = 0, .direction = 1}})});
  const ContinuousMeasurementContext plane_context(plane);
  ASSERT_EQ(plane_context.hops().size(), 4U);
  EXPECT_EQ(plane_context.seam_positions()[0],
            plane_context.layout().encode(std::array<std::size_t, 2>{3, 0}));
  EXPECT_EQ(plane_context.hops()[0].axis, 1U);
  EXPECT_EQ(plane_context.hops()[0].arrival,
            plane_context.layout().encode(std::array<std::size_t, 2>{3, 1}));
  EXPECT_EQ(plane_context.hops()[1].axis, 0U);
  EXPECT_EQ(plane_context.hops()[1].arrival,
            plane_context.layout().encode(std::array<std::size_t, 2>{2, 1}));
}

} // namespace
} // namespace qmc
