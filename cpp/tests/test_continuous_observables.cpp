#include "continuous_matsubara_detail.hpp"
#include "continuous_test_fixtures.hpp"
#include "lattice_transform_detail.hpp"
#include "qmc/continuous_observables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
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
static_assert(!std::default_initializable<ContinuousMatsubaraPlan>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousMatsubaraPlan &>().modes()),
                             const MatsubaraModeSet &>);
static_assert(!std::default_initializable<ContinuousParticleModes>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousParticleModes &>().model()),
                             const Model &>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousParticleModes &>().modes()),
                             const MatsubaraModeSet &>);

MatsubaraModeSet make_continuous_modes(const double beta, const TorusLayout &layout,
                                       std::vector<std::vector<std::size_t>> momenta,
                                       std::vector<std::int64_t> frequencies) {
  return MatsubaraModeSet(beta, layout,
                          MatsubaraModeRequest{.momentum_indices = std::move(momenta),
                                               .frequency_indices = std::move(frequencies)});
}

ContinuousConfiguration multidimensional_projection_configuration() {
  const Model model(ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 5,
      .dimension = 2,
      .hopping = 0.8,
  });
  const ContinuousPath first(1.0, {0, 0}, {0, 0},
                             {{.time = 0.0, .axis = 0, .direction = 1},
                              {.time = 0.2, .axis = 1, .direction = 1},
                              {.time = 0.2, .axis = 0, .direction = -1},
                              {.time = 0.55, .axis = 1, .direction = -1},
                              {.time = 1.0, .axis = 0, .direction = 1},
                              {.time = 1.0, .axis = 0, .direction = -1}});
  const ContinuousPath second(1.0, {2, 1}, {2, 1},
                              {{.time = 0.1, .axis = 1, .direction = -1},
                               {.time = 0.35, .axis = 0, .direction = -1},
                               {.time = 0.7, .axis = 1, .direction = 1},
                               {.time = 0.9, .axis = 0, .direction = 1}});
  return ContinuousConfiguration(model, Permutation({0, 1}), {first, second});
}

TEST(ContinuousMatsubaraPlanTest, OwnsModesAndEnforcesSymmetricFrequencyBound) {
  constexpr std::int64_t maximum = ContinuousMatsubaraPlan::kMaximumAbsoluteFrequencyIndex;
  const MatsubaraModeSet modes =
      make_continuous_modes(1.25, TorusLayout(5, 2), {{0, 0}, {4, 2}}, {-maximum, 0, maximum});
  const ContinuousMatsubaraPlan plan(modes);

  EXPECT_EQ(plan.modes(), modes);
  EXPECT_THROW(static_cast<void>(ContinuousMatsubaraPlan(
                   make_continuous_modes(1.25, TorusLayout(5, 2), {{0, 0}}, {maximum + 1}))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ContinuousMatsubaraPlan(
                   make_continuous_modes(1.25, TorusLayout(5, 2), {{0, 0}}, {-maximum - 1}))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ContinuousMatsubaraPlan(make_continuous_modes(
                   1.25, TorusLayout(5, 2), {{0, 0}}, {std::numeric_limits<std::int64_t>::min()}))),
               std::invalid_argument);
}

TEST(ContinuousMatsubaraPlanTest, ReducesMaximumIndexPhasesAndPreservesExactSeams) {
  constexpr std::int64_t maximum = ContinuousMatsubaraPlan::kMaximumAbsoluteFrequencyIndex;
  const std::complex<double> one{1.0, 0.0};
  EXPECT_EQ(detail::matsubara_time_phase(maximum, 0.0, 1.0), one);
  EXPECT_EQ(detail::matsubara_time_phase(maximum, 1.0, 1.0), one);
  EXPECT_EQ(detail::matsubara_time_phase(-maximum, 1.0, 1.0), one);

  const double half_cycle_time = std::ldexp(1.0, -21);
  const auto negative_one = detail::matsubara_time_phase(maximum, half_cycle_time, 1.0);
  EXPECT_NEAR(negative_one.real(), -1.0, 2e-15);
  EXPECT_NEAR(negative_one.imag(), 0.0, 2e-15);

  const double three_quarter_cycle_time = 3.0 * std::ldexp(1.0, -22);
  const auto negative_i = detail::matsubara_time_phase(maximum, three_quarter_cycle_time, 1.0);
  EXPECT_NEAR(negative_i.real(), 0.0, 2e-15);
  EXPECT_NEAR(negative_i.imag(), -1.0, 2e-15);
  EXPECT_NEAR(std::abs(detail::matsubara_time_phase(-maximum, three_quarter_cycle_time, 1.0) -
                       std::conj(negative_i)),
              0.0, 2e-15);
}

TEST(ContinuousMatsubaraPlanTest, IntervalKernelHandlesExactAndSmallArguments) {
  EXPECT_EQ(detail::matsubara_interval_transform(0, 0.25, 0.75, 1.0),
            std::complex<double>(0.5, 0.0));
  EXPECT_EQ(detail::matsubara_interval_transform(7, 0.0, 1.0, 1.0), std::complex<double>(0.0, 0.0));
  EXPECT_EQ(detail::matsubara_interval_transform(-7, 0.0, 1.0, 1.0),
            std::complex<double>(0.0, 0.0));
  EXPECT_EQ(detail::matsubara_interval_transform(3, 0.4, 0.4, 1.0), std::complex<double>(0.0, 0.0));

  const double begin = 0.4;
  const double end = std::nextafter(begin, 1.0);
  const double duration = end - begin;
  const auto small = detail::matsubara_interval_transform(1, begin, end, 1.0);
  const auto midpoint_phase = detail::matsubara_time_phase(1, begin + (duration / 2.0), 1.0);
  EXPECT_NEAR(std::abs((small / duration) - midpoint_phase), 0.0, 2e-15);

  const double omega = 4.0 * std::numbers::pi;
  const auto direct = (std::exp(std::complex<double>(0.0, omega * 0.73)) -
                       std::exp(std::complex<double>(0.0, omega * 0.12))) /
                      std::complex<double>(0.0, omega);
  EXPECT_NEAR(std::abs(detail::matsubara_interval_transform(2, 0.12, 0.73, 1.0) - direct), 0.0,
              5e-16);

  EXPECT_THROW(static_cast<void>(detail::matsubara_time_phase(1, -0.1, 1.0)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(detail::matsubara_interval_transform(1, 0.5, 0.4, 1.0)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(detail::matsubara_interval_transform(1, 0.0, 1.1, 1.0)),
               std::invalid_argument);
}

TEST(ContinuousMatsubaraPlanTest, SitePhasesMatchRetainedConventionWithoutIntegerOverflow) {
  const TorusLayout layout(4, 2);
  const MatsubaraModeSet modes = make_continuous_modes(1.0, layout, {{0, 0}, {3, 2}, {1, 3}}, {0});
  for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
    for (std::size_t site = 0; site < layout.volume(); ++site) {
      const auto components = layout.decode(SiteId(site));
      const double retained_phase =
          detail::phase_for_indices(modes.momentum_indices(momentum), components, layout);
      EXPECT_NEAR(std::abs(detail::matsubara_site_phase(modes, momentum, SiteId(site)) -
                           std::polar(1.0, -retained_phase)),
                  0.0, 3e-15);
    }
  }

  constexpr Coord large_size = std::numeric_limits<Coord>::max();
  const TorusLayout large_layout(large_size, 1);
  const auto large_component = static_cast<std::size_t>(large_size - 1);
  const MatsubaraModeSet large_modes =
      make_continuous_modes(1.0, large_layout, {{large_component}}, {0});
  const auto phase = detail::matsubara_site_phase(large_modes, 0, SiteId(large_component));
  const double reduced_angle = -2.0 * std::numbers::pi / static_cast<double>(large_size);
  EXPECT_NEAR(std::abs(phase - std::polar(1.0, reduced_angle)), 0.0, 1e-30);
}

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

TEST(ContinuousParticleModesTest, ProjectsStaticPathsAndCanonicalZeroMomentumExactly) {
  const Model model(ModelParameters{
      .particle_count = 2,
      .beta = 1.5,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 0.75,
  });
  const ContinuousConfiguration configuration(
      model, Permutation({0, 1}),
      {ContinuousPath(1.5, {0}, {0}, {}), ContinuousPath(1.5, {1}, {1}, {})});
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(4, 1), {{0}, {1}, {3}}, {-1, 0, 1});
  const ContinuousMatsubaraPlan plan(modes);
  const ContinuousParticleModes values = continuous_particle_modes(configuration, plan);

  EXPECT_EQ(values.model(), model);
  EXPECT_EQ(values.modes(), modes);
  for (std::size_t frequency : {std::size_t{0}, std::size_t{2}}) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      EXPECT_EQ(values.density(frequency, momentum), std::complex<double>(0.0, 0.0));
    }
  }
  EXPECT_EQ(values.density(1, 0), std::complex<double>(3.0, 0.0));
  const std::complex<double> expected_positive =
      model.beta() * (std::complex<double>(1.0, 0.0) + std::complex<double>(0.0, -1.0));
  EXPECT_NEAR(std::abs(values.density(1, 1) - expected_positive), 0.0, 2e-15);
  EXPECT_NEAR(std::abs(values.density(1, 2) - std::conj(expected_positive)), 0.0, 2e-15);
  EXPECT_EQ(values.axis_event_count(0), 0U);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      EXPECT_EQ(values.flux(frequency, momentum, 0), std::complex<double>(0.0, 0.0));
    }
  }

  EXPECT_THROW(static_cast<void>(values.density(3, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(values.density(0, 3)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(values.flux(3, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(values.flux(0, 3, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(values.flux(0, 0, 1)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(values.axis_event_count(1)), std::out_of_range);
}

TEST(ContinuousParticleModesTest, ZeroModeFluxIsExactCoveringDisplacement) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 2,
      .hopping = 1.0,
  });
  const ContinuousPath winding(1.0, {0, 0}, {3, -3},
                               {{.time = 0.0, .axis = 0, .direction = 1},
                                {.time = 0.1, .axis = 1, .direction = -1},
                                {.time = 0.2, .axis = 0, .direction = 1},
                                {.time = 0.4, .axis = 1, .direction = -1},
                                {.time = 0.8, .axis = 0, .direction = 1},
                                {.time = 1.0, .axis = 1, .direction = -1}});
  const ContinuousConfiguration configuration(model, Permutation({0}), {winding});
  const ContinuousMatsubaraPlan plan(
      make_continuous_modes(model.beta(), TorusLayout(3, 2), {{0, 0}}, {0}));
  const ContinuousParticleModes values = continuous_particle_modes(configuration, plan);
  const Site winding_number = configuration.total_winding();

  ASSERT_EQ(winding_number, Site({1, -1}));
  EXPECT_EQ(values.density(0, 0), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(values.flux(0, 0, 0),
            std::complex<double>(model.linear_size() * winding_number[0], 0.0));
  EXPECT_EQ(values.flux(0, 0, 1),
            std::complex<double>(model.linear_size() * winding_number[1], 0.0));
  EXPECT_EQ(values.axis_event_count(0), 3U);
  EXPECT_EQ(values.axis_event_count(1), 3U);
}

TEST(ContinuousParticleModesTest, WardIdentityIncludesCoincidentAndSeamEventGroups) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const MatsubaraModeSet modes =
      make_continuous_modes(configuration.model().beta(), TorusLayout(5, 1),
                            {{0}, {1}, {2}, {3}, {4}}, {-2, -1, 0, 1, 2});
  const ContinuousMatsubaraPlan plan(modes);
  const ContinuousMeasurementContext context(configuration);
  const ContinuousParticleModes values = continuous_particle_modes(context, plan);
  const ContinuousParticleModes convenience = continuous_particle_modes(configuration, plan);

  EXPECT_EQ(values.axis_event_count(0), configuration.event_count());
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      EXPECT_EQ(convenience.density(frequency, momentum), values.density(frequency, momentum));
      EXPECT_EQ(convenience.flux(frequency, momentum, 0), values.flux(frequency, momentum, 0));
      const std::complex<double> left =
          modes.frequency(frequency) * values.density(frequency, momentum);
      const std::complex<double> right = 2.0 *
                                         std::sin(0.5 * modes.wavevector_component(momentum, 0)) *
                                         values.flux(frequency, momentum, 0);
      const double tolerance = 3e-12 * (1.0 + std::abs(left) + std::abs(right));
      EXPECT_NEAR(std::abs(left - right), 0.0, tolerance);
    }
  }
}

TEST(ContinuousParticleModesTest, WardIdentityAndConjugationHoldInMultipleDimensions) {
  const ContinuousConfiguration configuration = multidimensional_projection_configuration();
  const TorusLayout layout(5, 2);
  const MatsubaraModeSet ward_modes = make_continuous_modes(
      configuration.model().beta(), layout, {{0, 0}, {1, 2}, {4, 3}, {2, 0}}, {-3, -1, 0, 2});
  const ContinuousParticleModes ward =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(ward_modes));
  for (std::size_t frequency = 0; frequency < ward_modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < ward_modes.momentum_count(); ++momentum) {
      std::complex<double> right{0.0, 0.0};
      for (std::size_t axis = 0; axis < layout.dimension(); ++axis) {
        right += 2.0 * std::sin(0.5 * ward_modes.wavevector_component(momentum, axis)) *
                 ward.flux(frequency, momentum, axis);
      }
      const std::complex<double> left =
          ward_modes.frequency(frequency) * ward.density(frequency, momentum);
      const double tolerance = 3e-12 * (1.0 + std::abs(left) + std::abs(right));
      EXPECT_NEAR(std::abs(left - right), 0.0, tolerance);
    }
  }

  const MatsubaraModeSet conjugate_modes = make_continuous_modes(
      configuration.model().beta(), layout, {{1, 2}, {4, 3}, {0, 2}, {0, 3}}, {-2, 2});
  const ContinuousParticleModes conjugate =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(conjugate_modes));
  EXPECT_NEAR(std::abs(std::conj(conjugate.density(0, 0)) - conjugate.density(1, 1)), 0.0, 2e-13);
  EXPECT_NEAR(std::abs(std::conj(conjugate.density(0, 1)) - conjugate.density(1, 0)), 0.0, 2e-13);
  EXPECT_NEAR(std::abs(std::conj(conjugate.density(0, 2)) - conjugate.density(1, 3)), 0.0, 2e-13);
  EXPECT_NEAR(std::abs(std::conj(conjugate.density(0, 3)) - conjugate.density(1, 2)), 0.0, 2e-13);
  for (const auto [momentum, opposite] :
       {std::pair{std::size_t{0}, std::size_t{1}}, std::pair{std::size_t{2}, std::size_t{3}}}) {
    for (std::size_t axis = 0; axis < layout.dimension(); ++axis) {
      // Midpoint bond fields are antiperiodic under q_alpha -> q_alpha+2*pi.
      // Canonicalizing -q into [0,L) therefore contributes one minus sign
      // when this flux component has nonzero momentum along its bond axis.
      const double reciprocal_gauge =
          conjugate_modes.momentum_indices(momentum)[axis] == 0 ? 1.0 : -1.0;
      EXPECT_NEAR(std::abs(reciprocal_gauge * std::conj(conjugate.flux(0, momentum, axis)) -
                           conjugate.flux(1, opposite, axis)),
                  0.0, 2e-13);
    }
  }
}

TEST(ContinuousParticleModesTest, TimeOriginRotationAppliesTheSignedMatsubaraPhase) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const double shift = 0.25;
  const ContinuousConfiguration rotated = rotate_configuration_time_origin(configuration, shift);
  const MatsubaraModeSet modes = make_continuous_modes(
      configuration.model().beta(), TorusLayout(5, 1), {{1}, {4}}, {-2, -1, 0, 1, 2});
  const ContinuousMatsubaraPlan plan(modes);
  const ContinuousParticleModes original_values = continuous_particle_modes(configuration, plan);
  const ContinuousParticleModes rotated_values = continuous_particle_modes(rotated, plan);

  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    const std::complex<double> rotation = std::conj(
        detail::matsubara_time_phase(modes.frequency_index(frequency), shift, modes.beta()));
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      EXPECT_NEAR(std::abs(rotated_values.density(frequency, momentum) -
                           rotation * original_values.density(frequency, momentum)),
                  0.0, 3e-13);
      EXPECT_NEAR(std::abs(rotated_values.flux(frequency, momentum, 0) -
                           rotation * original_values.flux(frequency, momentum, 0)),
                  0.0, 3e-13);
    }
  }
}

TEST(ContinuousParticleModesTest, RejectsContextAndPlanGeometryMismatch) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const ContinuousMeasurementContext context(configuration);
  const ContinuousMatsubaraPlan wrong_beta(
      make_continuous_modes(2.0, TorusLayout(5, 1), {{0}}, {0}));
  const ContinuousMatsubaraPlan wrong_layout(
      make_continuous_modes(1.0, TorusLayout(4, 1), {{0}}, {0}));

  EXPECT_THROW(static_cast<void>(continuous_particle_modes(context, wrong_beta)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(continuous_particle_modes(context, wrong_layout)),
               std::invalid_argument);
}

} // namespace
} // namespace qmc
