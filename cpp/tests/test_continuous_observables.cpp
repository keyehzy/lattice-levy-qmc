#include "continuous_matsubara_detail.hpp"
#include "continuous_test_fixtures.hpp"
#include "lattice_transform_detail.hpp"
#include "qmc/continuous_observables.hpp"
#include "qmc/interaction.hpp"

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
static_assert(!std::default_initializable<ContinuousDensityLagPlan>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousDensityLagPlan &>().lags()),
                             const ImaginaryTimeLagSet &>);
static_assert(!std::default_initializable<ContinuousDensityLagValues>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousDensityLagValues &>().model()),
                             const Model &>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousDensityLagValues &>().lags()),
                             const ImaginaryTimeLagSet &>);
static_assert(!std::default_initializable<ContinuousParticleModes>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousParticleModes &>().model()),
                             const Model &>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousParticleModes &>().modes()),
                             const MatsubaraModeSet &>);
static_assert(!std::default_initializable<ContinuousPairDensityModes>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousPairDensityModes &>().model()),
                             const Model &>);
static_assert(std::is_same_v<decltype(std::declval<const ContinuousPairDensityModes &>().modes()),
                             const MatsubaraModeSet &>);
static_assert(!std::default_initializable<ContinuousMatsubaraDensityCorrelations>);
static_assert(
    std::is_same_v<decltype(std::declval<const ContinuousMatsubaraDensityCorrelations &>().model()),
                   const Model &>);
static_assert(
    std::is_same_v<decltype(std::declval<const ContinuousMatsubaraDensityCorrelations &>().modes()),
                   const MatsubaraModeSet &>);
static_assert(!std::default_initializable<DensityMatsubaraAccumulator>);
static_assert(!std::default_initializable<DensityMatsubaraBlockSeries>);
static_assert(std::is_same_v<decltype(std::declval<const DensityMatsubaraBlockSeries &>().model()),
                             const Model &>);
static_assert(std::is_same_v<decltype(std::declval<const DensityMatsubaraBlockSeries &>().modes()),
                             const MatsubaraModeSet &>);
static_assert(!std::default_initializable<DensityMatsubaraBlockAccumulator>);
static_assert(!std::default_initializable<DensityLagBlockSeries>);
static_assert(
    std::is_same_v<decltype(std::declval<const DensityLagBlockSeries &>().model()), const Model &>);
static_assert(std::is_same_v<decltype(std::declval<const DensityLagBlockSeries &>().lags()),
                             const ImaginaryTimeLagSet &>);
static_assert(!std::default_initializable<DensityLagBlockAccumulator>);
static_assert(!std::default_initializable<HoppingResponse>);
static_assert(
    std::is_same_v<decltype(std::declval<const HoppingResponse &>().model()), const Model &>);
static_assert(std::is_same_v<decltype(std::declval<const HoppingResponse &>().modes()),
                             const MatsubaraModeSet &>);
static_assert(!std::default_initializable<HoppingResponseAccumulator>);
static_assert(!std::default_initializable<HoppingResponseBlockSeries>);
static_assert(std::is_same_v<decltype(std::declval<const HoppingResponseBlockSeries &>().model()),
                             const Model &>);
static_assert(std::is_same_v<decltype(std::declval<const HoppingResponseBlockSeries &>().modes()),
                             const MatsubaraModeSet &>);
static_assert(!std::default_initializable<HoppingResponseBlockAccumulator>);

MatsubaraModeSet make_continuous_modes(const double beta, const TorusLayout &layout,
                                       std::vector<std::vector<std::size_t>> momenta,
                                       std::vector<std::int64_t> frequencies) {
  return MatsubaraModeSet(beta, layout,
                          MatsubaraModeRequest{.momentum_indices = std::move(momenta),
                                               .frequency_indices = std::move(frequencies)});
}

ImaginaryTimeLagSet make_continuous_lags(const double beta, const TorusLayout &layout,
                                         std::vector<std::vector<std::size_t>> momenta,
                                         std::vector<double> lags) {
  return ImaginaryTimeLagSet(
      beta, layout,
      ImaginaryTimeLagRequest{.momentum_indices = std::move(momenta), .lags = std::move(lags)});
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

ContinuousConfiguration winding_projection_configuration() {
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
  return ContinuousConfiguration(model, Permutation({0}), {winding});
}

ContinuousConfiguration atomic_pair_density_configuration() {
  const Model model(ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousPath first(1.0, {0}, {1},
                             {{.time = 0.0, .axis = 0, .direction = 1},
                              {.time = 0.5, .axis = 0, .direction = -1},
                              {.time = 1.0, .axis = 0, .direction = 1}});
  const ContinuousPath second(1.0, {1}, {0},
                              {{.time = 0.0, .axis = 0, .direction = -1},
                               {.time = 0.5, .axis = 0, .direction = 1},
                               {.time = 1.0, .axis = 0, .direction = -1}});
  return ContinuousConfiguration(model, Permutation({1, 0}), {first, second});
}

ContinuousConfiguration static_density_lag_configuration() {
  const Model model(ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 1.0,
  });
  return ContinuousConfiguration(
      model, Permutation({0, 1}),
      {ContinuousPath(1.0, {0}, {0}, {}), ContinuousPath(1.0, {1}, {1}, {})});
}

ContinuousConfiguration single_event_density_lag_configuration() {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 1,
      .dimension = 1,
      .hopping = 1.0,
  });
  return ContinuousConfiguration(
      model, Permutation({0}),
      {ContinuousPath(1.0, {0}, {1}, {{.time = 0.4, .axis = 0, .direction = 1}})});
}

ContinuousConfiguration multiple_cycle_density_lag_configuration() {
  const Model model(ModelParameters{
      .particle_count = 3,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousPath first(1.0, {0}, {1}, {{.time = 0.3, .axis = 0, .direction = 1}});
  const ContinuousPath second(1.0, {1}, {0}, {{.time = 0.7, .axis = 0, .direction = -1}});
  const ContinuousPath third(1.0, {2}, {2}, {});
  return ContinuousConfiguration(model, Permutation({1, 0, 2}), {first, second, third});
}

ContinuousConfiguration translated_configuration(const ContinuousConfiguration &configuration,
                                                 const Site &displacement) {
  std::vector<ContinuousPath> translated_worldlines;
  translated_worldlines.reserve(configuration.worldlines().size());
  for (const ContinuousPath &path : configuration.worldlines()) {
    translated_worldlines.push_back(path.translated(displacement));
  }
  return ContinuousConfiguration(configuration.model(), configuration.topology(),
                                 std::move(translated_worldlines));
}

std::complex<double> reference_density_at(const ContinuousConfiguration &configuration,
                                          const ImaginaryTimeLagSet &lags,
                                          const std::size_t momentum, const double tau) {
  const TorusLayout &layout = lags.layout();
  std::complex<double> density{0.0, 0.0};
  for (const Site &position : configuration.positions_at(tau)) {
    const std::vector<std::size_t> components = layout.decode(layout.encode_covering(position));
    const double phase =
        detail::phase_for_indices(lags.momentum_indices(momentum), components, layout);
    density += std::polar(1.0, -phase);
  }
  return density;
}

double reference_raw_density_lag_overlap(const ContinuousConfiguration &configuration,
                                         const ImaginaryTimeLagSet &lags,
                                         const std::size_t momentum, const double tau) {
  const double beta = lags.beta();
  const ContinuousMeasurementContext context(configuration);
  std::vector<double> boundaries{0.0, beta};
  const auto append_boundary = [&boundaries, beta](double boundary) {
    if (boundary == beta) {
      boundary = 0.0;
    }
    if (boundary > 0.0 && boundary < beta) {
      boundaries.push_back(boundary);
    }
  };
  for (std::size_t group = 0; group < context.event_group_count(); ++group) {
    double event_time = context.event_time(group);
    append_boundary(event_time);
    if (event_time == beta) {
      event_time = 0.0;
    }
    double shifted = event_time - tau;
    if (shifted < 0.0) {
      shifted += beta;
    }
    append_boundary(shifted);
  }
  std::ranges::sort(boundaries);
  boundaries.erase(std::ranges::unique(boundaries).begin(), boundaries.end());

  double overlap = 0.0;
  for (std::size_t interval = 0; interval + 1 < boundaries.size(); ++interval) {
    const double begin = boundaries[interval];
    const double end = boundaries[interval + 1];
    const double midpoint = begin + (0.5 * (end - begin));
    double shifted_midpoint = midpoint + tau;
    if (shifted_midpoint >= beta) {
      shifted_midpoint -= beta;
    }
    const std::complex<double> base = reference_density_at(configuration, lags, momentum, midpoint);
    const std::complex<double> shifted =
        reference_density_at(configuration, lags, momentum, shifted_midpoint);
    overlap += (end - begin) * (shifted * std::conj(base)).real();
  }
  return overlap;
}

double reference_symmetrized_density_lag_overlap(const ContinuousConfiguration &configuration,
                                                 const ImaginaryTimeLagSet &lags,
                                                 const std::size_t momentum, const double tau) {
  const double reflected = tau == 0.0 ? 0.0 : lags.beta() - tau;
  return (0.5 * reference_raw_density_lag_overlap(configuration, lags, momentum, tau)) +
         (0.5 * reference_raw_density_lag_overlap(configuration, lags, momentum, reflected));
}

std::vector<double>
density_lag_linear_segment_boundaries(const ContinuousMeasurementContext &context) {
  const double beta = context.model().beta();
  std::vector<double> event_times{0.0};
  event_times.reserve(context.event_group_count() + 1);
  for (std::size_t group = 0; group < context.event_group_count(); ++group) {
    const double time = context.event_time(group);
    event_times.push_back(time == beta ? 0.0 : time);
  }
  std::ranges::sort(event_times);
  event_times.erase(std::ranges::unique(event_times).begin(), event_times.end());

  std::vector<double> boundaries{0.0, beta};
  boundaries.reserve((event_times.size() * event_times.size()) + 2);
  // A circular autocorrelation of step functions is affine between pairwise
  // differences of their discontinuity times.
  for (const double shifted_event : event_times) {
    for (const double base_event : event_times) {
      double boundary = shifted_event - base_event;
      if (boundary < 0.0) {
        boundary += beta;
      }
      if (boundary > 0.0 && boundary < beta) {
        boundaries.push_back(boundary);
      }
    }
  }
  std::ranges::sort(boundaries);

  const double merge_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * beta;
  std::vector<double> merged;
  merged.reserve(boundaries.size());
  for (const double boundary : boundaries) {
    if (merged.empty() || boundary - merged.back() > merge_tolerance) {
      merged.push_back(boundary);
    }
  }
  merged.front() = 0.0;
  merged.back() = beta;
  return merged;
}

std::complex<double> integrate_affine_matsubara_segment(const double begin, const double end,
                                                        const double value_at_begin,
                                                        const double slope,
                                                        const double frequency) {
  const double duration = end - begin;
  if (frequency == 0.0) {
    return {value_at_begin * duration + 0.5 * slope * duration * duration, 0.0};
  }

  const std::complex<double> imaginary{0.0, 1.0};
  const double phase_span = frequency * duration;
  std::complex<double> constant_integral;
  std::complex<double> linear_integral;
  if (std::abs(phase_span) < 1e-4) {
    std::complex<double> term{1.0, 0.0};
    std::complex<double> constant_series{1.0, 0.0};
    std::complex<double> linear_series{0.5, 0.0};
    for (std::size_t order = 1; order <= 12; ++order) {
      term *= imaginary * phase_span / static_cast<double>(order);
      constant_series += term / static_cast<double>(order + 1);
      linear_series += term / static_cast<double>(order + 2);
    }
    constant_integral = duration * constant_series;
    linear_integral = duration * duration * linear_series;
  } else {
    const std::complex<double> end_phase = std::polar(1.0, phase_span);
    constant_integral = (end_phase - 1.0) / (imaginary * frequency);
    linear_integral = (end_phase * (1.0 - imaginary * phase_span) - 1.0) / (frequency * frequency);
  }

  return std::polar(1.0, frequency * begin) *
         ((value_at_begin * constant_integral) + (slope * linear_integral));
}

std::complex<double> exact_piecewise_lag_transform(const ContinuousDensityLagValues &values,
                                                   const std::span<const double> segment_boundaries,
                                                   const std::size_t momentum,
                                                   const double frequency) {
  std::complex<double> transform{0.0, 0.0};
  for (std::size_t segment = 0; segment + 1 < segment_boundaries.size(); ++segment) {
    const double begin = segment_boundaries[segment];
    const double end = segment_boundaries[segment + 1];
    const double duration = end - begin;
    const double first_lag = begin + duration / 3.0;
    const double second_lag = begin + 2.0 * duration / 3.0;
    const double first_value = values.overlap(2 * segment, momentum);
    const double second_value = values.overlap((2 * segment) + 1, momentum);
    const double slope = (second_value - first_value) / (second_lag - first_lag);
    const double value_at_begin = first_value - slope * (first_lag - begin);
    transform += integrate_affine_matsubara_segment(begin, end, value_at_begin, slope, frequency);
  }
  return transform;
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

TEST(ContinuousDensityLagValuesTest, OwnsPlanAndResultProvenanceWithCheckedLagMajorAccess) {
  const ContinuousConfiguration configuration = static_density_lag_configuration();
  const Model model = configuration.model();
  const ImaginaryTimeLagSet lags =
      make_continuous_lags(model.beta(), TorusLayout(4, 1), {{0}, {1}, {3}}, {0.75, 0.0, 0.25});
  const ContinuousDensityLagPlan plan(lags);
  const ContinuousDensityLagValues from_context =
      continuous_density_lag_values(ContinuousMeasurementContext(configuration), plan);
  const ContinuousDensityLagValues convenience = continuous_density_lag_values(configuration, plan);

  EXPECT_EQ(plan.lags(), lags);
  EXPECT_EQ(from_context.model(), model);
  EXPECT_EQ(from_context.lags(), lags);
  for (std::size_t lag = 0; lag < lags.lag_count(); ++lag) {
    EXPECT_DOUBLE_EQ(from_context.overlap(lag, 0), 4.0);
    EXPECT_NEAR(from_context.overlap(lag, 1), 2.0, 2e-15);
    EXPECT_NEAR(from_context.overlap(lag, 2), 2.0, 2e-15);
    for (std::size_t momentum = 0; momentum < lags.momentum_count(); ++momentum) {
      EXPECT_DOUBLE_EQ(convenience.overlap(lag, momentum), from_context.overlap(lag, momentum));
    }
  }
  EXPECT_THROW(static_cast<void>(from_context.overlap(lags.lag_count(), 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(from_context.overlap(0, lags.momentum_count())),
               std::out_of_range);
}

TEST(ContinuousDensityLagValuesTest, MatchesIndependentIntervalReferenceAcrossEdgeFixtures) {
  std::vector<ContinuousConfiguration> configurations;
  configurations.push_back(static_density_lag_configuration());
  configurations.push_back(single_event_density_lag_configuration());
  configurations.push_back(test::coincident_seam_configuration());
  configurations.push_back(multidimensional_projection_configuration());
  configurations.push_back(multiple_cycle_density_lag_configuration());
  const Model empty_model(ModelParameters{
      .particle_count = 0,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  configurations.emplace_back(empty_model, Permutation(), std::vector<ContinuousPath>{});

  for (const ContinuousConfiguration &configuration : configurations) {
    const Model &model = configuration.model();
    const TorusLayout layout(model.linear_size(), model.dimension());
    std::vector<std::vector<std::size_t>> momenta(1,
                                                  std::vector<std::size_t>(model.dimension(), 0));
    if (model.linear_size() > 1) {
      std::vector<std::size_t> nonzero(model.dimension(), 0);
      nonzero[0] = 1;
      momenta.push_back(std::move(nonzero));
    }
    const ImaginaryTimeLagSet lags =
        make_continuous_lags(model.beta(), layout, std::move(momenta), {0.0, 0.13, 0.4, 0.73});
    const ContinuousDensityLagValues values =
        continuous_density_lag_values(configuration, ContinuousDensityLagPlan(lags));

    const double particle_count = static_cast<double>(model.particle_count());
    const double exact_zero_momentum = (model.beta() * particle_count) * particle_count;
    for (std::size_t lag = 0; lag < lags.lag_count(); ++lag) {
      EXPECT_DOUBLE_EQ(values.overlap(lag, 0), exact_zero_momentum);
      for (std::size_t momentum = 1; momentum < lags.momentum_count(); ++momentum) {
        const double expected =
            reference_symmetrized_density_lag_overlap(configuration, lags, momentum, lags.lag(lag));
        const double tolerance = 3e-13 * (1.0 + std::abs(expected));
        EXPECT_NEAR(values.overlap(lag, momentum), expected, tolerance);
      }
    }
  }
}

TEST(ContinuousDensityLagValuesTest,
     EnforcesTimeReflectionAndPreservesTranslationAndTimeOriginInvariance) {
  const ContinuousConfiguration configuration = multidimensional_projection_configuration();
  const ContinuousConfiguration translated = translated_configuration(configuration, Site{6, -7});
  const ContinuousConfiguration rotated = rotate_configuration_time_origin(configuration, 0.25);
  const Model model = configuration.model();
  const ImaginaryTimeLagSet lags = make_continuous_lags(
      model.beta(), TorusLayout(5, 2), {{0, 0}, {1, 2}, {4, 3}}, {0.0, 0.1, 0.25, 0.75, 0.9});
  const ContinuousDensityLagPlan plan(lags);
  const ContinuousDensityLagValues original = continuous_density_lag_values(configuration, plan);
  const ContinuousDensityLagValues shifted = continuous_density_lag_values(translated, plan);
  const ContinuousDensityLagValues time_rotated = continuous_density_lag_values(rotated, plan);

  const std::array<std::size_t, 5> reflected_indices{0, 4, 3, 2, 1};
  for (std::size_t lag = 0; lag < lags.lag_count(); ++lag) {
    for (std::size_t momentum = 0; momentum < lags.momentum_count(); ++momentum) {
      const double value = original.overlap(lag, momentum);
      const double tolerance = 5e-13 * (1.0 + std::abs(value));
      EXPECT_NEAR(value, original.overlap(reflected_indices[lag], momentum), tolerance);
      EXPECT_NEAR(value, shifted.overlap(lag, momentum), tolerance);
      EXPECT_NEAR(value, time_rotated.overlap(lag, momentum), tolerance);
      if (lags.lag(lag) == 0.0) {
        EXPECT_GE(value, 0.0);
      }
    }
  }
}

TEST(ContinuousDensityLagValuesTest,
     ExactPiecewiseLagTransformReproducesContinuousMatsubaraDensityObservation) {
  const std::array configurations{
      static_density_lag_configuration(),
      test::coincident_seam_configuration(),
      multidimensional_projection_configuration(),
      multiple_cycle_density_lag_configuration(),
  };

  for (const ContinuousConfiguration &configuration : configurations) {
    const Model &model = configuration.model();
    const TorusLayout layout(model.linear_size(), model.dimension());
    std::vector<std::vector<std::size_t>> momenta(1,
                                                  std::vector<std::size_t>(model.dimension(), 0));
    std::vector<std::size_t> forward(model.dimension(), 0);
    forward[0] = 1;
    momenta.push_back(forward);
    forward[0] = static_cast<std::size_t>(model.linear_size() - 1);
    momenta.push_back(forward);
    if (model.dimension() > 1) {
      std::vector<std::size_t> mixed(model.dimension(), 1);
      mixed[0] = 2;
      momenta.push_back(std::move(mixed));
    }

    const ContinuousMeasurementContext context(configuration);
    const std::vector<double> boundaries = density_lag_linear_segment_boundaries(context);
    std::vector<double> evaluation_lags;
    evaluation_lags.reserve(2 * (boundaries.size() - 1));
    for (std::size_t segment = 0; segment + 1 < boundaries.size(); ++segment) {
      const double begin = boundaries[segment];
      const double duration = boundaries[segment + 1] - begin;
      evaluation_lags.push_back(begin + duration / 3.0);
      evaluation_lags.push_back(begin + 2.0 * duration / 3.0);
    }

    const ImaginaryTimeLagSet lags =
        make_continuous_lags(model.beta(), layout, momenta, std::move(evaluation_lags));
    const ContinuousDensityLagValues lag_values =
        continuous_density_lag_values(context, ContinuousDensityLagPlan(lags));
    const MatsubaraModeSet modes =
        make_continuous_modes(model.beta(), layout, momenta, {-2, -1, 0, 1, 2});
    const ContinuousParticleModes mode_values =
        continuous_particle_modes(context, ContinuousMatsubaraPlan(modes));
    const double normalization = model.beta() * static_cast<double>(model.volume());

    for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
      for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
        const std::size_t reflected_frequency = modes.frequency_count() - frequency - 1;
        ASSERT_EQ(modes.frequency_index(reflected_frequency), -modes.frequency_index(frequency));
        const auto centered_norm = [&](const std::size_t frequency_ordinal) {
          std::complex<double> amplitude = mode_values.density(frequency_ordinal, momentum);
          if (momentum == 0 && modes.frequency_index(frequency_ordinal) == 0) {
            amplitude -= model.beta() * static_cast<double>(model.particle_count());
          }
          return std::norm(amplitude);
        };
        const double expected =
            0.5 * (centered_norm(frequency) + centered_norm(reflected_frequency)) / normalization;
        std::complex<double> integrated = exact_piecewise_lag_transform(
            lag_values, boundaries, momentum, modes.frequency(frequency));
        if (momentum == 0) {
          const double particle_count = static_cast<double>(model.particle_count());
          integrated -= integrate_affine_matsubara_segment(
              0.0, model.beta(), model.beta() * particle_count * particle_count, 0.0,
              modes.frequency(frequency));
        }
        integrated /= normalization;
        const double tolerance = 2e-11 * (1.0 + expected);
        EXPECT_NEAR(integrated.real(), expected, tolerance)
            << "frequency index = " << modes.frequency_index(frequency)
            << ", momentum ordinal = " << momentum;
        EXPECT_NEAR(integrated.imag(), 0.0, tolerance)
            << "frequency index = " << modes.frequency_index(frequency)
            << ", momentum ordinal = " << momentum;
      }
    }
  }
}

TEST(ContinuousDensityLagValuesTest, HandlesRepresentableLagsAdjacentToThePeriodicSeam) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const Model model = configuration.model();
  const double first_positive = std::nextafter(0.0, 1.0);
  const double last_before_beta = std::nextafter(model.beta(), 0.0);
  const ImaginaryTimeLagSet lags = make_continuous_lags(model.beta(), TorusLayout(5, 1), {{1}, {4}},
                                                        {0.0, first_positive, last_before_beta});
  const ContinuousDensityLagValues values =
      continuous_density_lag_values(configuration, ContinuousDensityLagPlan(lags));

  for (std::size_t lag = 0; lag < lags.lag_count(); ++lag) {
    for (std::size_t momentum = 0; momentum < lags.momentum_count(); ++momentum) {
      EXPECT_TRUE(std::isfinite(values.overlap(lag, momentum)));
    }
  }
  for (std::size_t momentum = 0; momentum < lags.momentum_count(); ++momentum) {
    EXPECT_NEAR(values.overlap(0, momentum), values.overlap(1, momentum), 2e-14);
    EXPECT_NEAR(values.overlap(0, momentum), values.overlap(2, momentum), 2e-14);
  }
}

TEST(ContinuousDensityLagValuesTest, RejectsEveryContextAndPlanGeometryMismatch) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const ContinuousMeasurementContext context(configuration);
  const ContinuousDensityLagPlan wrong_beta(
      make_continuous_lags(2.0, TorusLayout(5, 1), {{0}, {1}}, {0.0, 0.5}));
  const ContinuousDensityLagPlan wrong_layout(
      make_continuous_lags(1.0, TorusLayout(4, 1), {{0}, {1}}, {0.0, 0.5}));

  EXPECT_THROW(static_cast<void>(continuous_density_lag_values(context, wrong_beta)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(continuous_density_lag_values(context, wrong_layout)),
               std::invalid_argument);
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
  const ContinuousConfiguration configuration = winding_projection_configuration();
  const Model model = configuration.model();
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

TEST(ContinuousParticleModesTest,
     GlobalCoveringTranslationAppliesMomentumPhaseAndPreservesAutoResponses) {
  const ContinuousConfiguration configuration = multidimensional_projection_configuration();
  const Site displacement{6, -7};
  const ContinuousConfiguration translated = translated_configuration(configuration, displacement);
  const Model model = configuration.model();
  const TorusLayout layout(model.linear_size(), model.dimension());
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), layout, {{0, 0}, {1, 2}, {4, 3}, {2, 0}}, {-3, -1, 0, 2});
  const ContinuousMatsubaraPlan plan(modes);
  const ContinuousParticleModes original_values = continuous_particle_modes(configuration, plan);
  const ContinuousParticleModes translated_values = continuous_particle_modes(translated, plan);

  DensityMatsubaraAccumulator original_density_accumulator(model, modes);
  DensityMatsubaraAccumulator translated_density_accumulator(model, modes);
  HoppingResponseAccumulator original_hopping_accumulator(model, modes);
  HoppingResponseAccumulator translated_hopping_accumulator(model, modes);
  original_density_accumulator.observe(original_values);
  translated_density_accumulator.observe(translated_values);
  original_hopping_accumulator.observe(original_values);
  translated_hopping_accumulator.observe(translated_values);
  const ContinuousMatsubaraDensityCorrelations original_density =
      original_density_accumulator.finish();
  const ContinuousMatsubaraDensityCorrelations translated_density =
      translated_density_accumulator.finish();
  const HoppingResponse original_hopping = original_hopping_accumulator.finish();
  const HoppingResponse translated_hopping = translated_hopping_accumulator.finish();

  const auto expect_complex_near = [](const std::complex<double> actual,
                                      const std::complex<double> expected) {
    const double tolerance = 5e-13 * (1.0 + std::abs(actual) + std::abs(expected));
    EXPECT_NEAR(std::abs(actual - expected), 0.0, tolerance);
  };
  const SiteId physical_displacement = layout.encode_covering(displacement);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      const std::complex<double> translation_phase =
          detail::matsubara_site_phase(modes, momentum, physical_displacement);
      expect_complex_near(translated_values.density(frequency, momentum),
                          translation_phase * original_values.density(frequency, momentum));
      expect_complex_near(translated_density.mean_amplitude(frequency, momentum),
                          translation_phase * original_density.mean_amplitude(frequency, momentum));
      EXPECT_NEAR(translated_density.at(frequency, momentum),
                  original_density.at(frequency, momentum), 5e-13);

      for (std::size_t left = 0; left < model.dimension(); ++left) {
        expect_complex_near(translated_values.flux(frequency, momentum, left),
                            translation_phase * original_values.flux(frequency, momentum, left));
        expect_complex_near(translated_hopping.mean_flux(frequency, momentum, left),
                            translation_phase *
                                original_hopping.mean_flux(frequency, momentum, left));
        for (std::size_t right = 0; right < model.dimension(); ++right) {
          expect_complex_near(translated_hopping.flux_response(frequency, momentum, left, right),
                              original_hopping.flux_response(frequency, momentum, left, right));
          expect_complex_near(translated_hopping.paramagnetic(frequency, momentum, left, right),
                              original_hopping.paramagnetic(frequency, momentum, left, right));
        }
      }
    }
  }
  for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
    EXPECT_EQ(translated_values.axis_event_count(axis), original_values.axis_event_count(axis));
    EXPECT_EQ(translated_hopping.diamagnetic(axis), original_hopping.diamagnetic(axis));
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

TEST(ContinuousPairDensityModesTest, ProjectsStaticOnsitePairsWithOwnedProvenance) {
  const Model model(ModelParameters{
      .particle_count = 3,
      .beta = 1.5,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 0.75,
  });
  const ContinuousConfiguration configuration(model, Permutation({0, 1, 2}),
                                              {ContinuousPath(model.beta(), {1}, {1}, {}),
                                               ContinuousPath(model.beta(), {1}, {1}, {}),
                                               ContinuousPath(model.beta(), {3}, {3}, {})});
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(4, 1), {{0}, {1}, {3}}, {-1, 0, 1});
  const ContinuousPairDensityModes values =
      continuous_pair_density_modes(configuration, ContinuousMatsubaraPlan(modes));

  EXPECT_EQ(values.model(), model);
  EXPECT_EQ(values.modes(), modes);
  for (const std::size_t frequency : {std::size_t{0}, std::size_t{2}}) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      EXPECT_EQ(values.pair_density(frequency, momentum), std::complex<double>(0.0, 0.0));
    }
  }
  EXPECT_EQ(values.pair_density(1, 0), std::complex<double>(model.beta(), 0.0));
  EXPECT_NEAR(std::abs(values.pair_density(1, 1) - std::complex<double>(0.0, -model.beta())), 0.0,
              2e-15);
  EXPECT_NEAR(std::abs(values.pair_density(1, 2) - std::complex<double>(0.0, model.beta())), 0.0,
              2e-15);

  EXPECT_THROW(static_cast<void>(values.pair_density(modes.frequency_count(), 0)),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(values.pair_density(0, modes.momentum_count())),
               std::out_of_range);
}

TEST(ContinuousPairDensityModesTest, IntegratesDynamicPairResidenceExactly) {
  const Model model(ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousConfiguration configuration(
      model, Permutation({0, 1}),
      {ContinuousPath(model.beta(), {2}, {2}, {}),
       ContinuousPath(model.beta(), {2}, {2},
                      {{.time = 0.25, .axis = 0, .direction = -1},
                       {.time = 0.75, .axis = 0, .direction = 1}})});
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}, {1}, {2}}, {-1, 0, 1});
  const ContinuousPairDensityModes values =
      continuous_pair_density_modes(configuration, ContinuousMatsubaraPlan(modes));

  ASSERT_DOUBLE_EQ(pair_overlap_time(configuration), 0.5);
  for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
    const std::complex<double> site_phase =
        detail::matsubara_site_phase(modes, momentum, SiteId(2));
    EXPECT_NEAR(std::abs(values.pair_density(0, momentum) - site_phase / std::numbers::pi), 0.0,
                2e-15);
    EXPECT_NEAR(std::abs(values.pair_density(1, momentum) - 0.5 * site_phase), 0.0, 2e-15);
    EXPECT_NEAR(std::abs(values.pair_density(2, momentum) - site_phase / std::numbers::pi), 0.0,
                2e-15);
  }
}

TEST(ContinuousPairDensityModesTest, EqualTimeSwapsNeverExposeTransientOccupancies) {
  const ContinuousConfiguration configuration = atomic_pair_density_configuration();
  const Model model = configuration.model();
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}, {1}, {2}}, {-2, -1, 0, 1, 2});
  const ContinuousPairDensityModes values =
      continuous_pair_density_modes(configuration, ContinuousMatsubaraPlan(modes));

  ASSERT_EQ(ContinuousMeasurementContext(configuration).event_group_count(), 3U);
  EXPECT_EQ(pair_overlap_time(configuration), 0.0);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      EXPECT_EQ(values.pair_density(frequency, momentum), std::complex<double>(0.0, 0.0));
    }
  }
}

TEST(ContinuousPairDensityModesTest, ZeroModeExactlyMatchesSharedPairOverlapSweep) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const Model model = configuration.model();
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(5, 1), {{0}, {1}, {4}}, {-2, -1, 0, 1, 2});
  const ContinuousMatsubaraPlan plan(modes);
  const ContinuousMeasurementContext context(configuration);
  const ContinuousPairDensityModes from_context = continuous_pair_density_modes(context, plan);
  const ContinuousPairDensityModes convenience = continuous_pair_density_modes(configuration, plan);

  ASSERT_DOUBLE_EQ(pair_overlap_time(configuration), 0.5);
  EXPECT_EQ(from_context.pair_density(2, 0),
            std::complex<double>(pair_overlap_time(configuration), 0.0));
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      EXPECT_EQ(convenience.pair_density(frequency, momentum),
                from_context.pair_density(frequency, momentum));
    }
  }
}

TEST(ContinuousPairDensityModesTest, ObeysSiteFieldConjugationAndTimeOriginRotation) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const double shift = 0.25;
  const ContinuousConfiguration rotated = rotate_configuration_time_origin(configuration, shift);
  const MatsubaraModeSet modes = make_continuous_modes(
      configuration.model().beta(), TorusLayout(5, 1), {{0}, {1}, {4}}, {-2, -1, 0, 1, 2});
  const ContinuousMatsubaraPlan plan(modes);
  const ContinuousPairDensityModes original = continuous_pair_density_modes(configuration, plan);
  const ContinuousPairDensityModes shifted = continuous_pair_density_modes(rotated, plan);
  const std::array<std::size_t, 5> opposite_frequency{4, 3, 2, 1, 0};
  const std::array<std::size_t, 3> opposite_momentum{0, 2, 1};

  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    const std::complex<double> rotation = std::conj(
        detail::matsubara_time_phase(modes.frequency_index(frequency), shift, modes.beta()));
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      EXPECT_NEAR(std::abs(std::conj(original.pair_density(frequency, momentum)) -
                           original.pair_density(opposite_frequency[frequency],
                                                 opposite_momentum[momentum])),
                  0.0, 3e-13);
      EXPECT_NEAR(std::abs(shifted.pair_density(frequency, momentum) -
                           rotation * original.pair_density(frequency, momentum)),
                  0.0, 3e-13);
    }
  }
}

TEST(ContinuousPairDensityModesTest, GlobalCoveringTranslationAppliesMomentumPhase) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const Site displacement{7};
  const ContinuousConfiguration translated = translated_configuration(configuration, displacement);
  const TorusLayout layout(configuration.model().linear_size(), configuration.model().dimension());
  const MatsubaraModeSet modes = make_continuous_modes(configuration.model().beta(), layout,
                                                       {{0}, {1}, {4}}, {-2, -1, 0, 1, 2});
  const ContinuousMatsubaraPlan plan(modes);
  const ContinuousPairDensityModes original = continuous_pair_density_modes(configuration, plan);
  const ContinuousPairDensityModes shifted = continuous_pair_density_modes(translated, plan);

  ASSERT_GT(std::abs(original.pair_density(2, 1)), 0.1);
  const SiteId physical_displacement = layout.encode_covering(displacement);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      const std::complex<double> translation_phase =
          detail::matsubara_site_phase(modes, momentum, physical_displacement);
      const std::complex<double> expected =
          translation_phase * original.pair_density(frequency, momentum);
      const std::complex<double> actual = shifted.pair_density(frequency, momentum);
      const double tolerance = 5e-13 * (1.0 + std::abs(actual) + std::abs(expected));
      EXPECT_NEAR(std::abs(actual - expected), 0.0, tolerance);
    }
  }
}

TEST(ContinuousPairDensityModesTest, RejectsContextAndPlanGeometryMismatch) {
  const ContinuousConfiguration configuration = test::coincident_seam_configuration();
  const ContinuousMeasurementContext context(configuration);
  const ContinuousMatsubaraPlan wrong_beta(
      make_continuous_modes(2.0, TorusLayout(5, 1), {{0}}, {0}));
  const ContinuousMatsubaraPlan wrong_layout(
      make_continuous_modes(1.0, TorusLayout(4, 1), {{0}}, {0}));

  EXPECT_THROW(static_cast<void>(continuous_pair_density_modes(context, wrong_beta)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(continuous_pair_density_modes(context, wrong_layout)),
               std::invalid_argument);
}

TEST(ContinuousPairDensityModesTest, HandlesEmptyAndSingleSiteConfigurations) {
  const Model empty_model(ModelParameters{
      .particle_count = 0,
      .beta = 1.25,
      .linear_size = 1,
      .dimension = 2,
      .hopping = 0.8,
  });
  const MatsubaraModeSet empty_modes =
      make_continuous_modes(empty_model.beta(), TorusLayout(1, 2), {{0, 0}}, {-1, 0, 1});
  const ContinuousPairDensityModes empty =
      continuous_pair_density_modes(ContinuousConfiguration(empty_model, Permutation(), {}),
                                    ContinuousMatsubaraPlan(empty_modes));
  for (std::size_t frequency = 0; frequency < empty_modes.frequency_count(); ++frequency) {
    EXPECT_EQ(empty.pair_density(frequency, 0), std::complex<double>(0.0, 0.0));
  }

  const Model single_site_model(ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 1,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousConfiguration single_site(
      single_site_model, Permutation({0, 1}),
      {ContinuousPath(
           single_site_model.beta(), {0}, {0},
           {{.time = 0.25, .axis = 0, .direction = 1}, {.time = 0.75, .axis = 0, .direction = -1}}),
       ContinuousPath(single_site_model.beta(), {0}, {0}, {})});
  const MatsubaraModeSet single_site_modes =
      make_continuous_modes(single_site_model.beta(), TorusLayout(1, 1), {{0}}, {-1, 0, 1});
  const ContinuousPairDensityModes values =
      continuous_pair_density_modes(single_site, ContinuousMatsubaraPlan(single_site_modes));

  EXPECT_NEAR(std::abs(values.pair_density(0, 0)), 0.0, 1e-15);
  EXPECT_EQ(values.pair_density(1, 0), std::complex<double>(1.0, 0.0));
  EXPECT_NEAR(std::abs(values.pair_density(2, 0)), 0.0, 1e-15);
}

TEST(ContinuousPairDensityModesTest, RejectsANonFiniteProjectedZeroMode) {
  const Model model(ModelParameters{
      .particle_count = 3,
      .beta = std::numeric_limits<double>::max() * 0.75,
      .linear_size = 1,
      .dimension = 1,
      .hopping = 0.0,
  });
  const ContinuousConfiguration configuration(model, Permutation({0, 1, 2}),
                                              {ContinuousPath(model.beta(), {0}, {0}, {}),
                                               ContinuousPath(model.beta(), {0}, {0}, {}),
                                               ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousMatsubaraPlan plan(
      make_continuous_modes(model.beta(), TorusLayout(1, 1), {{0}}, {0}));

  EXPECT_THROW(static_cast<void>(continuous_pair_density_modes(configuration, plan)),
               std::overflow_error);
}

TEST(DensityMatsubaraAccumulatorTest, UsesAnalyticCenteringAndPublishesCompleteResultProvenance) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 2.0,
      .linear_size = 4,
      .dimension = 1,
      .hopping = 0.75,
  });
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(4, 1), {{0}, {1}}, {0, 1});
  const ContinuousMatsubaraPlan plan(modes);
  const ContinuousConfiguration first(model, Permutation({0}),
                                      {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousConfiguration second(model, Permutation({0}),
                                       {ContinuousPath(model.beta(), {2}, {2}, {})});
  const ContinuousParticleModes first_values = continuous_particle_modes(first, plan);
  const ContinuousParticleModes second_values = continuous_particle_modes(second, plan);
  DensityMatsubaraAccumulator accumulator(model, modes);

  EXPECT_EQ(accumulator.model(), model);
  EXPECT_EQ(accumulator.modes(), modes);
  EXPECT_EQ(accumulator.sample_count(), 0U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);

  accumulator.observe(first_values);
  const ContinuousMatsubaraDensityCorrelations first_result = accumulator.finish();
  EXPECT_EQ(first_result.model(), model);
  EXPECT_EQ(first_result.modes(), modes);
  EXPECT_EQ(first_result.sample_count(), 1U);
  EXPECT_EQ(first_result.mean_amplitude(0, 0), std::complex<double>(2.0, 0.0));
  EXPECT_EQ(first_result.at(0, 0), 0.0);
  EXPECT_NEAR(std::abs(first_result.mean_amplitude(0, 1) - std::complex<double>(2.0, 0.0)), 0.0,
              1e-15);
  EXPECT_NEAR(first_result.at(0, 1), 0.5, 2e-15);
  EXPECT_EQ(first_result.mean_amplitude(1, 0), std::complex<double>(0.0, 0.0));
  EXPECT_EQ(first_result.mean_amplitude(1, 1), std::complex<double>(0.0, 0.0));
  EXPECT_EQ(first_result.at(1, 0), 0.0);
  EXPECT_EQ(first_result.at(1, 1), 0.0);

  accumulator.observe(second_values);
  const ContinuousMatsubaraDensityCorrelations result = accumulator.finish();
  EXPECT_EQ(accumulator.sample_count(), 2U);
  EXPECT_EQ(result.sample_count(), 2U);
  EXPECT_EQ(result.mean_amplitude(0, 0), std::complex<double>(2.0, 0.0));
  EXPECT_EQ(result.at(0, 0), 0.0);
  EXPECT_NEAR(std::abs(result.mean_amplitude(0, 1)), 0.0, 2e-15);
  EXPECT_NEAR(result.at(0, 1), 0.5, 2e-15);

  EXPECT_THROW(static_cast<void>(result.mean_amplitude(2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(result.mean_amplitude(0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(result.at(2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(result.at(0, 2)), std::out_of_range);
}

TEST(DensityMatsubaraAccumulatorTest, RejectsEveryProvenanceMismatchBeforeMutation) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}, {1}}, {0});
  EXPECT_THROW(static_cast<void>(DensityMatsubaraAccumulator(
                   model, make_continuous_modes(2.0, TorusLayout(3, 1), {{0}}, {0}))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(DensityMatsubaraAccumulator(
                   model, make_continuous_modes(1.0, TorusLayout(4, 1), {{0}}, {0}))),
               std::invalid_argument);

  const ContinuousConfiguration configuration(model, Permutation({0}),
                                              {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  const Model different_particle_model(ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousConfiguration different_particle_configuration(
      different_particle_model, Permutation({0, 1}),
      {ContinuousPath(different_particle_model.beta(), {0}, {0}, {}),
       ContinuousPath(different_particle_model.beta(), {1}, {1}, {})});
  const ContinuousParticleModes different_particle_values =
      continuous_particle_modes(different_particle_configuration, ContinuousMatsubaraPlan(modes));
  const Model different_model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 2.0,
  });
  const ContinuousConfiguration different_configuration(
      different_model, Permutation({0}), {ContinuousPath(different_model.beta(), {0}, {0}, {})});
  const ContinuousParticleModes different_model_values =
      continuous_particle_modes(different_configuration, ContinuousMatsubaraPlan(modes));
  const MatsubaraModeSet different_modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}, {1}}, {1});
  const ContinuousParticleModes different_mode_values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(different_modes));
  DensityMatsubaraAccumulator accumulator(model, modes);

  EXPECT_THROW(accumulator.observe(different_particle_values), std::invalid_argument);
  EXPECT_EQ(accumulator.sample_count(), 0U);
  EXPECT_THROW(accumulator.observe(different_model_values), std::invalid_argument);
  EXPECT_EQ(accumulator.sample_count(), 0U);
  EXPECT_THROW(accumulator.observe(different_mode_values), std::invalid_argument);
  EXPECT_EQ(accumulator.sample_count(), 0U);

  accumulator.observe(values);
  EXPECT_EQ(accumulator.sample_count(), 1U);
  EXPECT_EQ(accumulator.finish().mean_amplitude(0, 0), std::complex<double>(1.0, 0.0));
}

TEST(DensityMatsubaraAccumulatorTest, OverflowLeavesAllPreviouslyObservedMomentsUnchanged) {
  const double beta = std::numeric_limits<double>::max() * 0.75;
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = beta,
      .linear_size = 1,
      .dimension = 1,
      .hopping = 0.0,
  });
  const MatsubaraModeSet modes = make_continuous_modes(model.beta(), TorusLayout(1, 1), {{0}}, {0});
  const ContinuousConfiguration configuration(model, Permutation({0}),
                                              {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  DensityMatsubaraAccumulator accumulator(model, modes);

  accumulator.observe(values);
  const ContinuousMatsubaraDensityCorrelations before = accumulator.finish();
  ASSERT_EQ(before.mean_amplitude(0, 0), std::complex<double>(beta, 0.0));
  ASSERT_EQ(before.at(0, 0), 0.0);

  EXPECT_THROW(accumulator.observe(values), std::overflow_error);
  EXPECT_EQ(accumulator.sample_count(), 1U);
  const ContinuousMatsubaraDensityCorrelations after = accumulator.finish();
  EXPECT_EQ(after.mean_amplitude(0, 0), before.mean_amplitude(0, 0));
  EXPECT_EQ(after.at(0, 0), before.at(0, 0));
  EXPECT_EQ(after.sample_count(), before.sample_count());
}

TEST(DensityMatsubaraAccumulatorTest, HandlesTheEmptyFixedParticleEnsemble) {
  const Model model(ModelParameters{
      .particle_count = 0,
      .beta = 1.25,
      .linear_size = 3,
      .dimension = 2,
      .hopping = 0.8,
  });
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 2), {{0, 0}, {1, 2}}, {-1, 0, 1});
  const ContinuousConfiguration configuration(model, Permutation(), {});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  DensityMatsubaraAccumulator accumulator(model, modes);

  accumulator.observe(values);
  const ContinuousMatsubaraDensityCorrelations result = accumulator.finish();
  EXPECT_EQ(result.sample_count(), 1U);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      EXPECT_EQ(result.mean_amplitude(frequency, momentum), std::complex<double>(0.0, 0.0));
      EXPECT_EQ(result.at(frequency, momentum), 0.0);
    }
  }
}

TEST(DensityMatsubaraBlockAccumulatorTest,
     ReproducesKnownBlockStatisticsAndTheSimpleAccumulatorMean) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  });
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(2, 1), {{0}, {1}}, {0, 1});
  const ContinuousMatsubaraPlan plan(modes);
  const ContinuousConfiguration static_configuration(model, Permutation({0}),
                                                     {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousConfiguration moving_configuration(
      model, Permutation({0}),
      {ContinuousPath(model.beta(), {0}, {0},
                      {{.time = 0.25, .axis = 0, .direction = 1},
                       {.time = 0.75, .axis = 0, .direction = -1}})});
  const ContinuousParticleModes static_values =
      continuous_particle_modes(static_configuration, plan);
  const ContinuousParticleModes moving_values =
      continuous_particle_modes(moving_configuration, plan);
  DensityMatsubaraAccumulator simple_accumulator(model, modes);
  DensityMatsubaraBlockAccumulator block_accumulator(model, modes, 2);

  EXPECT_EQ(block_accumulator.model(), model);
  EXPECT_EQ(block_accumulator.modes(), modes);
  EXPECT_EQ(block_accumulator.measurements_per_block(), 2U);
  EXPECT_EQ(block_accumulator.completed_block_count(), 0U);
  EXPECT_EQ(block_accumulator.pending_sample_count(), 0U);
  EXPECT_THROW(static_cast<void>(block_accumulator.finish()), std::logic_error);

  const auto observe = [&](const ContinuousParticleModes &values) {
    simple_accumulator.observe(values);
    block_accumulator.observe(values);
  };
  observe(static_values);
  EXPECT_EQ(block_accumulator.completed_block_count(), 0U);
  EXPECT_EQ(block_accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(static_cast<void>(block_accumulator.finish()), std::logic_error);
  observe(static_values);
  EXPECT_EQ(block_accumulator.completed_block_count(), 1U);
  EXPECT_EQ(block_accumulator.pending_sample_count(), 0U);
  EXPECT_THROW(static_cast<void>(block_accumulator.finish()), std::logic_error);

  observe(static_values);
  EXPECT_EQ(block_accumulator.completed_block_count(), 1U);
  EXPECT_EQ(block_accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(static_cast<void>(block_accumulator.finish()), std::logic_error);
  observe(moving_values);
  observe(moving_values);
  observe(moving_values);

  const DensityMatsubaraBlockSeries series = block_accumulator.finish();
  const ContinuousMatsubaraDensityCorrelations simple = simple_accumulator.finish();
  ASSERT_EQ(series.model(), model);
  ASSERT_EQ(series.modes(), modes);
  ASSERT_EQ(series.measurements_per_block(), 2U);
  ASSERT_EQ(series.block_count(), 3U);
  ASSERT_EQ(series.sample_count(), 6U);

  const double inverse_pi_squared = 1.0 / (std::numbers::pi * std::numbers::pi);
  EXPECT_NEAR(series.block_value(0, 0, 1), 0.5, 1e-15);
  EXPECT_NEAR(series.block_value(0, 1, 1), 0.0, 1e-15);
  EXPECT_NEAR(series.block_value(1, 0, 1), 0.25, 1e-15);
  EXPECT_NEAR(series.block_value(1, 1, 1), inverse_pi_squared, 2e-15);
  EXPECT_NEAR(series.block_value(2, 0, 1), 0.0, 1e-15);
  EXPECT_NEAR(series.block_value(2, 1, 1), 2.0 * inverse_pi_squared, 3e-15);
  EXPECT_NEAR(series.mean(0, 1), 0.25, 1e-15);
  EXPECT_NEAR(series.mean(1, 1), inverse_pi_squared, 2e-15);

  const double expected_frequency_zero_variance = 1.0 / 48.0;
  const double expected_frequency_one_variance = 1.0 / (3.0 * std::pow(std::numbers::pi, 4));
  const double expected_covariance = -1.0 / (12.0 * std::numbers::pi * std::numbers::pi);
  EXPECT_NEAR(series.covariance_of_mean(1, 0, 0), expected_frequency_zero_variance, 1e-15);
  EXPECT_NEAR(series.covariance_of_mean(1, 1, 1), expected_frequency_one_variance, 2e-16);
  EXPECT_NEAR(series.covariance_of_mean(1, 0, 1), expected_covariance, 3e-16);
  EXPECT_EQ(series.covariance_of_mean(1, 0, 1), series.covariance_of_mean(1, 1, 0));
  EXPECT_NEAR(series.standard_error(0, 1), std::sqrt(expected_frequency_zero_variance), 1e-15);
  EXPECT_NEAR(series.standard_error(1, 1), std::sqrt(expected_frequency_one_variance), 1e-15);

  EXPECT_NEAR(series.jackknife_mean(0, 0, 1), 0.125, 1e-15);
  EXPECT_NEAR(series.jackknife_mean(0, 1, 1), 1.5 * inverse_pi_squared, 2e-15);
  EXPECT_NEAR(series.jackknife_mean(1, 0, 1), 0.25, 1e-15);
  EXPECT_NEAR(series.jackknife_mean(1, 1, 1), inverse_pi_squared, 2e-15);
  EXPECT_NEAR(series.jackknife_mean(2, 0, 1), 0.375, 1e-15);
  EXPECT_NEAR(series.jackknife_mean(2, 1, 1), 0.5 * inverse_pi_squared, 2e-15);

  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    EXPECT_EQ(series.block_value(0, frequency, 0), 0.0);
    EXPECT_EQ(series.block_value(1, frequency, 0), 0.0);
    EXPECT_EQ(series.block_value(2, frequency, 0), 0.0);
    EXPECT_EQ(series.mean(frequency, 0), 0.0);
    EXPECT_EQ(series.standard_error(frequency, 0), 0.0);
    EXPECT_EQ(series.mean(frequency, 0), simple.at(frequency, 0));
    EXPECT_NEAR(series.mean(frequency, 1), simple.at(frequency, 1), 2e-15);
  }

  EXPECT_THROW(static_cast<void>(series.block_value(3, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.block_value(0, 2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.block_value(0, 0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.mean(2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.mean(0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.covariance_of_mean(2, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.covariance_of_mean(0, 2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.covariance_of_mean(0, 0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.standard_error(2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.standard_error(0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.jackknife_mean(3, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.jackknife_mean(0, 2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.jackknife_mean(0, 0, 2)), std::out_of_range);
}

TEST(DensityMatsubaraBlockAccumulatorTest, RejectsInvalidConstructionAndProvenanceBeforeMutation) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}, {1}}, {0, 1});
  EXPECT_THROW(static_cast<void>(DensityMatsubaraBlockAccumulator(model, modes, 0)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(DensityMatsubaraBlockAccumulator(
                   model, make_continuous_modes(2.0, TorusLayout(3, 1), {{0}}, {0}), 2)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(DensityMatsubaraBlockAccumulator(
                   model, make_continuous_modes(1.0, TorusLayout(4, 1), {{0}}, {0}), 2)),
               std::invalid_argument);

  const ContinuousConfiguration configuration(model, Permutation({0}),
                                              {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  const Model different_model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 2.0,
  });
  const ContinuousConfiguration different_configuration(
      different_model, Permutation({0}), {ContinuousPath(different_model.beta(), {0}, {0}, {})});
  const ContinuousParticleModes different_model_values =
      continuous_particle_modes(different_configuration, ContinuousMatsubaraPlan(modes));
  const MatsubaraModeSet different_modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}, {1}}, {1});
  const ContinuousParticleModes different_mode_values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(different_modes));
  DensityMatsubaraBlockAccumulator accumulator(model, modes, 2);

  accumulator.observe(values);
  ASSERT_EQ(accumulator.completed_block_count(), 0U);
  ASSERT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(accumulator.observe(different_model_values), std::invalid_argument);
  EXPECT_EQ(accumulator.completed_block_count(), 0U);
  EXPECT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(accumulator.observe(different_mode_values), std::invalid_argument);
  EXPECT_EQ(accumulator.completed_block_count(), 0U);
  EXPECT_EQ(accumulator.pending_sample_count(), 1U);

  accumulator.observe(values);
  EXPECT_EQ(accumulator.completed_block_count(), 1U);
  EXPECT_EQ(accumulator.pending_sample_count(), 0U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);
  accumulator.observe(values);
  EXPECT_EQ(accumulator.completed_block_count(), 1U);
  EXPECT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);
  accumulator.observe(values);
  EXPECT_EQ(accumulator.completed_block_count(), 2U);
  EXPECT_EQ(accumulator.pending_sample_count(), 0U);
  EXPECT_NO_THROW(static_cast<void>(accumulator.finish()));
}

TEST(DensityMatsubaraBlockAccumulatorTest, PreservesRequestedFrequencyOrder) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  });
  const MatsubaraModeSet forward_modes =
      make_continuous_modes(model.beta(), TorusLayout(2, 1), {{1}}, {0, 1});
  const MatsubaraModeSet reverse_modes =
      make_continuous_modes(model.beta(), TorusLayout(2, 1), {{1}}, {1, 0});
  const ContinuousConfiguration static_configuration(model, Permutation({0}),
                                                     {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousConfiguration moving_configuration(
      model, Permutation({0}),
      {ContinuousPath(model.beta(), {0}, {0},
                      {{.time = 0.25, .axis = 0, .direction = 1},
                       {.time = 0.75, .axis = 0, .direction = -1}})});
  DensityMatsubaraBlockAccumulator forward(model, forward_modes, 1);
  DensityMatsubaraBlockAccumulator reverse(model, reverse_modes, 1);
  for (const ContinuousConfiguration *configuration :
       {&static_configuration, &moving_configuration, &static_configuration}) {
    forward.observe(
        continuous_particle_modes(*configuration, ContinuousMatsubaraPlan(forward_modes)));
    reverse.observe(
        continuous_particle_modes(*configuration, ContinuousMatsubaraPlan(reverse_modes)));
  }
  const DensityMatsubaraBlockSeries forward_series = forward.finish();
  const DensityMatsubaraBlockSeries reverse_series = reverse.finish();

  for (std::size_t block = 0; block < forward_series.block_count(); ++block) {
    EXPECT_NEAR(forward_series.block_value(block, 0, 0), reverse_series.block_value(block, 1, 0),
                1e-15);
    EXPECT_NEAR(forward_series.block_value(block, 1, 0), reverse_series.block_value(block, 0, 0),
                1e-15);
  }
  EXPECT_NEAR(forward_series.mean(0, 0), reverse_series.mean(1, 0), 1e-15);
  EXPECT_NEAR(forward_series.mean(1, 0), reverse_series.mean(0, 0), 1e-15);
  for (std::size_t left = 0; left < 2; ++left) {
    for (std::size_t right = 0; right < 2; ++right) {
      EXPECT_NEAR(forward_series.covariance_of_mean(0, left, right),
                  reverse_series.covariance_of_mean(0, 1 - left, 1 - right), 1e-15);
    }
  }
}

TEST(DensityLagBlockAccumulatorTest, ReproducesSignedBlockStatisticsAndInstallsExactZeroMomentum) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 2.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ImaginaryTimeLagSet lags =
      make_continuous_lags(model.beta(), TorusLayout(2, 1), {{0}, {1}}, {0.0, 1.0});
  const ContinuousDensityLagPlan plan(lags);
  const ContinuousConfiguration static_configuration(model, Permutation({0}),
                                                     {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousConfiguration moving_configuration(
      model, Permutation({0}),
      {ContinuousPath(
          model.beta(), {0}, {0},
          {{.time = 0.5, .axis = 0, .direction = 1}, {.time = 1.5, .axis = 0, .direction = -1}})});
  const ContinuousDensityLagValues static_values =
      continuous_density_lag_values(static_configuration, plan);
  const ContinuousDensityLagValues moving_values =
      continuous_density_lag_values(moving_configuration, plan);
  DensityLagBlockAccumulator accumulator(model, lags, 2);

  EXPECT_EQ(accumulator.model(), model);
  EXPECT_EQ(accumulator.lags(), lags);
  EXPECT_EQ(accumulator.measurements_per_block(), 2U);
  EXPECT_EQ(accumulator.completed_block_count(), 0U);
  EXPECT_EQ(accumulator.pending_sample_count(), 0U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);

  accumulator.observe(static_values);
  EXPECT_EQ(accumulator.completed_block_count(), 0U);
  EXPECT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);
  accumulator.observe(static_values);
  EXPECT_EQ(accumulator.completed_block_count(), 1U);
  EXPECT_EQ(accumulator.pending_sample_count(), 0U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);
  accumulator.observe(static_values);
  EXPECT_EQ(accumulator.completed_block_count(), 1U);
  EXPECT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);
  accumulator.observe(moving_values);
  accumulator.observe(moving_values);
  accumulator.observe(moving_values);

  const DensityLagBlockSeries series = accumulator.finish();
  ASSERT_EQ(series.model(), model);
  ASSERT_EQ(series.lags(), lags);
  ASSERT_EQ(series.measurements_per_block(), 2U);
  ASSERT_EQ(series.block_count(), 3U);
  ASSERT_EQ(series.sample_count(), 6U);

  for (std::size_t block = 0; block < series.block_count(); ++block) {
    for (std::size_t lag = 0; lag < lags.lag_count(); ++lag) {
      EXPECT_DOUBLE_EQ(series.block_value(block, lag, 0), 0.0);
    }
  }
  EXPECT_DOUBLE_EQ(series.mean(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(series.mean(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(series.standard_error(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(series.standard_error(1, 0), 0.0);

  EXPECT_NEAR(series.block_value(0, 0, 1), 0.5, 1e-15);
  EXPECT_NEAR(series.block_value(1, 0, 1), 0.5, 1e-15);
  EXPECT_NEAR(series.block_value(2, 0, 1), 0.5, 1e-15);
  EXPECT_NEAR(series.block_value(0, 1, 1), 0.5, 1e-15);
  EXPECT_NEAR(series.block_value(1, 1, 1), 0.0, 1e-15);
  EXPECT_NEAR(series.block_value(2, 1, 1), -0.5, 1e-15);
  EXPECT_NEAR(series.mean(0, 1), 0.5, 1e-15);
  EXPECT_NEAR(series.mean(1, 1), 0.0, 1e-15);

  EXPECT_NEAR(series.covariance_of_mean(1, 0, 0), 0.0, 1e-15);
  EXPECT_NEAR(series.covariance_of_mean(1, 1, 1), 1.0 / 12.0, 1e-15);
  EXPECT_NEAR(series.covariance_of_mean(1, 0, 1), 0.0, 1e-15);
  EXPECT_DOUBLE_EQ(series.covariance_of_mean(1, 0, 1), series.covariance_of_mean(1, 1, 0));
  EXPECT_NEAR(series.standard_error(0, 1), 0.0, 1e-15);
  EXPECT_NEAR(series.standard_error(1, 1), std::sqrt(1.0 / 12.0), 1e-15);

  EXPECT_NEAR(series.jackknife_mean(0, 1, 1), -0.25, 1e-15);
  EXPECT_NEAR(series.jackknife_mean(1, 1, 1), 0.0, 1e-15);
  EXPECT_NEAR(series.jackknife_mean(2, 1, 1), 0.25, 1e-15);

  EXPECT_THROW(static_cast<void>(series.block_value(3, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.block_value(0, 2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.block_value(0, 0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.mean(2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.mean(0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.covariance_of_mean(2, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.covariance_of_mean(0, 2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.covariance_of_mean(0, 0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.standard_error(2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.standard_error(0, 2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.jackknife_mean(3, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.jackknife_mean(0, 2, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.jackknife_mean(0, 0, 2)), std::out_of_range);
}

TEST(DensityLagBlockAccumulatorTest, RejectsInvalidConstructionAndProvenanceBeforeMutation) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ImaginaryTimeLagSet lags =
      make_continuous_lags(model.beta(), TorusLayout(3, 1), {{0}, {1}}, {0.0, 0.5});
  EXPECT_THROW(static_cast<void>(DensityLagBlockAccumulator(model, lags, 0)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(DensityLagBlockAccumulator(
                   model, make_continuous_lags(2.0, TorusLayout(3, 1), {{0}}, {0.0}), 2)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(DensityLagBlockAccumulator(
                   model, make_continuous_lags(1.0, TorusLayout(4, 1), {{0}}, {0.0}), 2)),
               std::invalid_argument);

  const ContinuousConfiguration configuration(model, Permutation({0}),
                                              {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousDensityLagValues values =
      continuous_density_lag_values(configuration, ContinuousDensityLagPlan(lags));
  const Model different_model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 2.0,
  });
  const ContinuousConfiguration different_configuration(
      different_model, Permutation({0}), {ContinuousPath(different_model.beta(), {0}, {0}, {})});
  const ContinuousDensityLagValues different_model_values =
      continuous_density_lag_values(different_configuration, ContinuousDensityLagPlan(lags));
  const ImaginaryTimeLagSet different_lags =
      make_continuous_lags(model.beta(), TorusLayout(3, 1), {{0}, {1}}, {0.25});
  const ContinuousDensityLagValues different_lag_values =
      continuous_density_lag_values(configuration, ContinuousDensityLagPlan(different_lags));
  DensityLagBlockAccumulator accumulator(model, lags, 2);

  accumulator.observe(values);
  ASSERT_EQ(accumulator.completed_block_count(), 0U);
  ASSERT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(accumulator.observe(different_model_values), std::invalid_argument);
  EXPECT_EQ(accumulator.completed_block_count(), 0U);
  EXPECT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(accumulator.observe(different_lag_values), std::invalid_argument);
  EXPECT_EQ(accumulator.completed_block_count(), 0U);
  EXPECT_EQ(accumulator.pending_sample_count(), 1U);

  accumulator.observe(values);
  EXPECT_EQ(accumulator.completed_block_count(), 1U);
  EXPECT_EQ(accumulator.pending_sample_count(), 0U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);
  accumulator.observe(values);
  EXPECT_EQ(accumulator.completed_block_count(), 1U);
  EXPECT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);
  accumulator.observe(values);
  EXPECT_EQ(accumulator.completed_block_count(), 2U);
  EXPECT_EQ(accumulator.pending_sample_count(), 0U);
  EXPECT_NO_THROW(static_cast<void>(accumulator.finish()));
}

TEST(DensityLagBlockAccumulatorTest, PreservesRequestedLagOrder) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ImaginaryTimeLagSet forward_lags =
      make_continuous_lags(model.beta(), TorusLayout(2, 1), {{1}}, {0.0, 0.5});
  const ImaginaryTimeLagSet reverse_lags =
      make_continuous_lags(model.beta(), TorusLayout(2, 1), {{1}}, {0.5, 0.0});
  const ContinuousConfiguration static_configuration(model, Permutation({0}),
                                                     {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousConfiguration moving_configuration(
      model, Permutation({0}),
      {ContinuousPath(model.beta(), {0}, {0},
                      {{.time = 0.25, .axis = 0, .direction = 1},
                       {.time = 0.75, .axis = 0, .direction = -1}})});
  DensityLagBlockAccumulator forward(model, forward_lags, 1);
  DensityLagBlockAccumulator reverse(model, reverse_lags, 1);
  for (const ContinuousConfiguration *configuration :
       {&static_configuration, &moving_configuration, &static_configuration}) {
    forward.observe(
        continuous_density_lag_values(*configuration, ContinuousDensityLagPlan(forward_lags)));
    reverse.observe(
        continuous_density_lag_values(*configuration, ContinuousDensityLagPlan(reverse_lags)));
  }
  const DensityLagBlockSeries forward_series = forward.finish();
  const DensityLagBlockSeries reverse_series = reverse.finish();

  for (std::size_t block = 0; block < forward_series.block_count(); ++block) {
    EXPECT_NEAR(forward_series.block_value(block, 0, 0), reverse_series.block_value(block, 1, 0),
                1e-15);
    EXPECT_NEAR(forward_series.block_value(block, 1, 0), reverse_series.block_value(block, 0, 0),
                1e-15);
  }
  EXPECT_NEAR(forward_series.mean(0, 0), reverse_series.mean(1, 0), 1e-15);
  EXPECT_NEAR(forward_series.mean(1, 0), reverse_series.mean(0, 0), 1e-15);
  for (std::size_t left = 0; left < 2; ++left) {
    for (std::size_t right = 0; right < 2; ++right) {
      EXPECT_NEAR(forward_series.covariance_of_mean(0, left, right),
                  reverse_series.covariance_of_mean(0, 1 - left, 1 - right), 1e-15);
    }
  }
}

TEST(HoppingResponseAccumulatorTest,
     UsesExactZeroMeanAndPublishesHermitianResponseWithCompleteProvenance) {
  const ContinuousConfiguration configuration = multidimensional_projection_configuration();
  const Model model = configuration.model();
  const MatsubaraModeSet modes = make_continuous_modes(
      model.beta(), TorusLayout(model.linear_size(), model.dimension()), {{0, 0}, {1, 2}}, {0, 1});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  HoppingResponseAccumulator accumulator(model, modes);

  EXPECT_EQ(accumulator.model(), model);
  EXPECT_EQ(accumulator.modes(), modes);
  EXPECT_EQ(accumulator.sample_count(), 0U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);

  accumulator.observe(values);
  accumulator.observe(values);
  const HoppingResponse result = accumulator.finish();
  ASSERT_EQ(accumulator.sample_count(), 2U);
  ASSERT_EQ(result.model(), model);
  ASSERT_EQ(result.modes(), modes);
  ASSERT_EQ(result.sample_count(), 2U);

  const double normalization = model.beta() * static_cast<double>(model.volume());
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      for (std::size_t left = 0; left < model.dimension(); ++left) {
        const std::complex<double> left_flux = values.flux(frequency, momentum, left);
        EXPECT_NEAR(std::abs(result.mean_flux(frequency, momentum, left) - left_flux), 0.0, 1e-15);
        for (std::size_t right = 0; right < model.dimension(); ++right) {
          const std::complex<double> expected_response =
              left_flux * std::conj(values.flux(frequency, momentum, right)) / normalization;
          EXPECT_NEAR(
              std::abs(result.flux_response(frequency, momentum, left, right) - expected_response),
              0.0, 2e-15);
          EXPECT_EQ(result.flux_response(frequency, momentum, right, left),
                    std::conj(result.flux_response(frequency, momentum, left, right)));
          const std::complex<double> expected_paramagnetic =
              (left == right ? std::complex<double>(result.diamagnetic(left), 0.0)
                             : std::complex<double>(0.0, 0.0)) -
              expected_response;
          EXPECT_NEAR(std::abs(result.paramagnetic(frequency, momentum, left, right) -
                               expected_paramagnetic),
                      0.0, 2e-15);
        }
      }
    }
  }
  for (std::size_t axis = 0; axis < model.dimension(); ++axis) {
    EXPECT_NEAR(result.diamagnetic(axis),
                static_cast<double>(values.axis_event_count(axis)) / normalization, 1e-15);
  }

  EXPECT_THROW(static_cast<void>(result.mean_flux(modes.frequency_count(), 0, 0)),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(result.mean_flux(0, modes.momentum_count(), 0)),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(result.mean_flux(0, 0, model.dimension())), std::out_of_range);
  EXPECT_THROW(static_cast<void>(result.flux_response(0, 0, model.dimension(), 0)),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(result.flux_response(0, 0, 0, model.dimension())),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(result.diamagnetic(model.dimension())), std::out_of_range);
  EXPECT_THROW(static_cast<void>(result.paramagnetic(modes.frequency_count(), 0, 0, 0)),
               std::out_of_range);
}

TEST(HoppingResponseBlockAccumulatorTest,
     ReproducesSimpleMeansAndBlockErrorsForEveryAuthoritativeTerm) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 2,
      .dimension = 1,
      .hopping = 1.0,
  });
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(2, 1), {{0}, {1}}, {0, 1});
  const ContinuousMatsubaraPlan plan(modes);
  const std::vector configurations{
      ContinuousConfiguration(model, Permutation({0}),
                              {ContinuousPath(model.beta(), {0}, {0}, {})}),
      ContinuousConfiguration(model, Permutation({0}),
                              {ContinuousPath(model.beta(), {0}, {0},
                                              {{.time = 0.25, .axis = 0, .direction = 1},
                                               {.time = 0.75, .axis = 0, .direction = -1}})}),
      ContinuousConfiguration(model, Permutation({0}),
                              {ContinuousPath(model.beta(), {0}, {2},
                                              {{.time = 0.25, .axis = 0, .direction = 1},
                                               {.time = 0.75, .axis = 0, .direction = 1}})}),
  };
  HoppingResponseAccumulator simple_accumulator(model, modes);
  HoppingResponseBlockAccumulator block_accumulator(model, modes, 1);
  std::vector<HoppingResponse> block_references;
  block_references.reserve(configurations.size());

  EXPECT_EQ(block_accumulator.model(), model);
  EXPECT_EQ(block_accumulator.modes(), modes);
  EXPECT_EQ(block_accumulator.measurements_per_block(), 1U);
  EXPECT_EQ(block_accumulator.completed_block_count(), 0U);
  EXPECT_EQ(block_accumulator.pending_sample_count(), 0U);
  EXPECT_THROW(static_cast<void>(block_accumulator.finish()), std::logic_error);
  for (const ContinuousConfiguration &configuration : configurations) {
    const ContinuousParticleModes values = continuous_particle_modes(configuration, plan);
    simple_accumulator.observe(values);
    block_accumulator.observe(values);
    HoppingResponseAccumulator reference_accumulator(model, modes);
    reference_accumulator.observe(values);
    block_references.push_back(reference_accumulator.finish());
  }

  const HoppingResponse simple = simple_accumulator.finish();
  const HoppingResponseBlockSeries series = block_accumulator.finish();
  ASSERT_EQ(series.model(), model);
  ASSERT_EQ(series.modes(), modes);
  ASSERT_EQ(series.measurements_per_block(), 1U);
  ASSERT_EQ(series.block_count(), configurations.size());
  ASSERT_EQ(series.sample_count(), configurations.size());
  ASSERT_EQ(block_accumulator.completed_block_count(), configurations.size());
  ASSERT_EQ(block_accumulator.pending_sample_count(), 0U);

  const auto real_standard_error = [](const std::vector<std::complex<double>> &values) {
    const double count = static_cast<double>(values.size());
    double mean = 0.0;
    for (const std::complex<double> value : values) {
      mean += value.real() / count;
    }
    double squared_difference_sum = 0.0;
    for (const std::complex<double> value : values) {
      squared_difference_sum += std::pow(value.real() - mean, 2);
    }
    return std::sqrt(squared_difference_sum / (count * (count - 1.0)));
  };
  const auto imaginary_standard_error = [](const std::vector<std::complex<double>> &values) {
    const double count = static_cast<double>(values.size());
    double mean = 0.0;
    for (const std::complex<double> value : values) {
      mean += value.imag() / count;
    }
    double squared_difference_sum = 0.0;
    for (const std::complex<double> value : values) {
      squared_difference_sum += std::pow(value.imag() - mean, 2);
    }
    return std::sqrt(squared_difference_sum / (count * (count - 1.0)));
  };

  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      std::vector<std::complex<double>> flux_blocks;
      std::vector<std::complex<double>> response_blocks;
      std::vector<std::complex<double>> paramagnetic_blocks;
      for (std::size_t block = 0; block < block_references.size(); ++block) {
        const HoppingResponse &reference = block_references[block];
        flux_blocks.push_back(reference.mean_flux(frequency, momentum, 0));
        response_blocks.push_back(reference.flux_response(frequency, momentum, 0, 0));
        paramagnetic_blocks.push_back(reference.paramagnetic(frequency, momentum, 0, 0));
        EXPECT_EQ(series.block_mean_flux(block, frequency, momentum, 0), flux_blocks.back());
        EXPECT_EQ(series.block_flux_response(block, frequency, momentum, 0, 0),
                  response_blocks.back());
        EXPECT_EQ(series.block_paramagnetic(block, frequency, momentum, 0, 0),
                  paramagnetic_blocks.back());
      }
      EXPECT_NEAR(std::abs(series.mean_flux(frequency, momentum, 0) -
                           simple.mean_flux(frequency, momentum, 0)),
                  0.0, 1e-15);
      EXPECT_NEAR(std::abs(series.flux_response(frequency, momentum, 0, 0) -
                           simple.flux_response(frequency, momentum, 0, 0)),
                  0.0, 1e-15);
      EXPECT_NEAR(std::abs(series.paramagnetic(frequency, momentum, 0, 0) -
                           simple.paramagnetic(frequency, momentum, 0, 0)),
                  0.0, 1e-15);
      EXPECT_NEAR(
          series.mean_flux_standard_error(frequency, momentum, 0, HoppingResponseComponent::Real),
          real_standard_error(flux_blocks), 1e-15);
      EXPECT_NEAR(series.mean_flux_standard_error(frequency, momentum, 0,
                                                  HoppingResponseComponent::Imaginary),
                  imaginary_standard_error(flux_blocks), 1e-15);
      EXPECT_NEAR(series.flux_response_standard_error(frequency, momentum, 0, 0,
                                                      HoppingResponseComponent::Real),
                  real_standard_error(response_blocks), 1e-15);
      EXPECT_NEAR(series.flux_response_standard_error(frequency, momentum, 0, 0,
                                                      HoppingResponseComponent::Imaginary),
                  imaginary_standard_error(response_blocks), 1e-15);
      EXPECT_NEAR(series.paramagnetic_standard_error(frequency, momentum, 0, 0,
                                                     HoppingResponseComponent::Real),
                  real_standard_error(paramagnetic_blocks), 1e-15);
      EXPECT_NEAR(series.paramagnetic_standard_error(frequency, momentum, 0, 0,
                                                     HoppingResponseComponent::Imaginary),
                  imaginary_standard_error(paramagnetic_blocks), 1e-15);
      for (std::size_t omitted = 0; omitted < block_references.size(); ++omitted) {
        const std::size_t left = (omitted + 1) % block_references.size();
        const std::size_t right = (omitted + 2) % block_references.size();
        EXPECT_NEAR(std::abs(series.jackknife_mean_flux(omitted, frequency, momentum, 0) -
                             ((flux_blocks[left] + flux_blocks[right]) / 2.0)),
                    0.0, 1e-15);
        EXPECT_NEAR(std::abs(series.jackknife_flux_response(omitted, frequency, momentum, 0, 0) -
                             ((response_blocks[left] + response_blocks[right]) / 2.0)),
                    0.0, 1e-15);
        EXPECT_NEAR(std::abs(series.jackknife_paramagnetic(omitted, frequency, momentum, 0, 0) -
                             ((paramagnetic_blocks[left] + paramagnetic_blocks[right]) / 2.0)),
                    0.0, 1e-15);
      }
    }
  }

  std::vector<double> diamagnetic_blocks;
  for (std::size_t block = 0; block < block_references.size(); ++block) {
    diamagnetic_blocks.push_back(block_references[block].diamagnetic(0));
    EXPECT_EQ(series.block_diamagnetic(block, 0), diamagnetic_blocks.back());
  }
  double diamagnetic_mean = 0.0;
  for (const double value : diamagnetic_blocks) {
    diamagnetic_mean += value / static_cast<double>(diamagnetic_blocks.size());
  }
  double diamagnetic_squared_difference_sum = 0.0;
  for (const double value : diamagnetic_blocks) {
    diamagnetic_squared_difference_sum += std::pow(value - diamagnetic_mean, 2);
  }
  const double diamagnetic_error = std::sqrt(diamagnetic_squared_difference_sum /
                                             (static_cast<double>(diamagnetic_blocks.size()) *
                                              static_cast<double>(diamagnetic_blocks.size() - 1)));
  EXPECT_NEAR(series.diamagnetic(0), simple.diamagnetic(0), 1e-15);
  EXPECT_NEAR(series.diamagnetic_standard_error(0), diamagnetic_error, 1e-15);
  for (std::size_t omitted = 0; omitted < block_references.size(); ++omitted) {
    const std::size_t left = (omitted + 1) % block_references.size();
    const std::size_t right = (omitted + 2) % block_references.size();
    EXPECT_NEAR(series.jackknife_diamagnetic(omitted, 0),
                (diamagnetic_blocks[left] + diamagnetic_blocks[right]) / 2.0, 1e-15);
  }

  EXPECT_THROW(static_cast<void>(series.block_mean_flux(3, 0, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.block_flux_response(0, 2, 0, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.block_paramagnetic(0, 0, 2, 0, 0)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.block_diamagnetic(0, 1)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(series.jackknife_diamagnetic(3, 0)), std::out_of_range);
}

TEST(HoppingResponseBlockAccumulatorTest,
     RejectsInvalidConstructionAndProvenanceWithoutLosingPendingSamples) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}, {1}}, {0});
  EXPECT_THROW(static_cast<void>(HoppingResponseBlockAccumulator(model, modes, 0)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(HoppingResponseBlockAccumulator(
                   model, make_continuous_modes(2.0, TorusLayout(3, 1), {{0}}, {0}), 2)),
               std::invalid_argument);

  const ContinuousConfiguration configuration(model, Permutation({0}),
                                              {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  const Model different_model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 2.0,
  });
  const ContinuousConfiguration different_configuration(
      different_model, Permutation({0}), {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousParticleModes different_values =
      continuous_particle_modes(different_configuration, ContinuousMatsubaraPlan(modes));
  HoppingResponseBlockAccumulator accumulator(model, modes, 2);
  accumulator.observe(values);
  ASSERT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_THROW(accumulator.observe(different_values), std::invalid_argument);
  EXPECT_EQ(accumulator.pending_sample_count(), 1U);
  EXPECT_EQ(accumulator.completed_block_count(), 0U);
  accumulator.observe(values);
  accumulator.observe(values);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::logic_error);
  accumulator.observe(values);
  EXPECT_NO_THROW(static_cast<void>(accumulator.finish()));
}

TEST(HoppingResponseAccumulatorTest, ZeroModeResponseMatchesWindingAndEventCountIdentities) {
  const ContinuousConfiguration configuration = winding_projection_configuration();
  const Model model = configuration.model();
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 2), {{0, 0}}, {0});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  HoppingResponseAccumulator accumulator(model, modes);
  accumulator.observe(values);
  const HoppingResponse result = accumulator.finish();
  const Site winding = configuration.total_winding();
  const double normalization = model.beta() * static_cast<double>(model.volume());

  for (std::size_t left = 0; left < model.dimension(); ++left) {
    const double left_flux =
        static_cast<double>(model.linear_size()) * static_cast<double>(winding[left]);
    EXPECT_EQ(result.mean_flux(0, 0, left), std::complex<double>(left_flux, 0.0));
    EXPECT_NEAR(result.diamagnetic(left),
                static_cast<double>(values.axis_event_count(left)) / normalization, 1e-15);
    for (std::size_t right = 0; right < model.dimension(); ++right) {
      const double right_flux =
          static_cast<double>(model.linear_size()) * static_cast<double>(winding[right]);
      EXPECT_NEAR(result.flux_response(0, 0, left, right).real(),
                  left_flux * right_flux / normalization, 1e-15);
      EXPECT_EQ(result.flux_response(0, 0, left, right).imag(), 0.0);
    }
  }
}

TEST(HoppingResponseAccumulatorTest, RejectsEveryProvenanceMismatchBeforeMutation) {
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}, {1}}, {0});
  EXPECT_THROW(static_cast<void>(HoppingResponseAccumulator(
                   model, make_continuous_modes(2.0, TorusLayout(3, 1), {{0}}, {0}))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(HoppingResponseAccumulator(
                   model, make_continuous_modes(1.0, TorusLayout(4, 1), {{0}}, {0}))),
               std::invalid_argument);

  const ContinuousConfiguration configuration(model, Permutation({0}),
                                              {ContinuousPath(model.beta(), {0}, {0}, {})});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  const Model different_particle_model(ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousConfiguration different_particle_configuration(
      different_particle_model, Permutation({0, 1}),
      {ContinuousPath(different_particle_model.beta(), {0}, {0}, {}),
       ContinuousPath(different_particle_model.beta(), {1}, {1}, {})});
  const ContinuousParticleModes different_particle_values =
      continuous_particle_modes(different_particle_configuration, ContinuousMatsubaraPlan(modes));
  const Model different_model(ModelParameters{
      .particle_count = 1,
      .beta = 1.0,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 2.0,
  });
  const ContinuousConfiguration different_configuration(
      different_model, Permutation({0}), {ContinuousPath(different_model.beta(), {0}, {0}, {})});
  const ContinuousParticleModes different_model_values =
      continuous_particle_modes(different_configuration, ContinuousMatsubaraPlan(modes));
  const MatsubaraModeSet different_modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}, {1}}, {1});
  const ContinuousParticleModes different_mode_values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(different_modes));
  HoppingResponseAccumulator accumulator(model, modes);

  EXPECT_THROW(accumulator.observe(different_particle_values), std::invalid_argument);
  EXPECT_EQ(accumulator.sample_count(), 0U);
  EXPECT_THROW(accumulator.observe(different_model_values), std::invalid_argument);
  EXPECT_EQ(accumulator.sample_count(), 0U);
  EXPECT_THROW(accumulator.observe(different_mode_values), std::invalid_argument);
  EXPECT_EQ(accumulator.sample_count(), 0U);

  accumulator.observe(values);
  EXPECT_EQ(accumulator.sample_count(), 1U);
  const HoppingResponse result = accumulator.finish();
  EXPECT_EQ(result.mean_flux(0, 0, 0), std::complex<double>(0.0, 0.0));
  EXPECT_EQ(result.flux_response(0, 0, 0, 0), std::complex<double>(0.0, 0.0));
  EXPECT_EQ(result.diamagnetic(0), 0.0);
  EXPECT_EQ(result.paramagnetic(0, 0, 0, 0), std::complex<double>(0.0, 0.0));
}

TEST(HoppingResponseAccumulatorTest, HandlesTheEmptyFixedParticleEnsemble) {
  const Model model(ModelParameters{
      .particle_count = 0,
      .beta = 1.25,
      .linear_size = 3,
      .dimension = 2,
      .hopping = 0.8,
  });
  const MatsubaraModeSet modes =
      make_continuous_modes(model.beta(), TorusLayout(3, 2), {{0, 0}, {1, 2}}, {-1, 0, 1});
  const ContinuousConfiguration configuration(model, Permutation(), {});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  HoppingResponseAccumulator accumulator(model, modes);

  accumulator.observe(values);
  const HoppingResponse result = accumulator.finish();
  ASSERT_EQ(result.sample_count(), 1U);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      for (std::size_t left = 0; left < model.dimension(); ++left) {
        EXPECT_EQ(result.mean_flux(frequency, momentum, left), std::complex<double>(0.0, 0.0));
        EXPECT_EQ(result.diamagnetic(left), 0.0);
        for (std::size_t right = 0; right < model.dimension(); ++right) {
          EXPECT_EQ(result.flux_response(frequency, momentum, left, right),
                    std::complex<double>(0.0, 0.0));
          EXPECT_EQ(result.paramagnetic(frequency, momentum, left, right),
                    std::complex<double>(0.0, 0.0));
        }
      }
    }
  }
}

TEST(HoppingResponseAccumulatorTest, RejectsNonFiniteNormalizationAtFinish) {
  const double beta = std::numeric_limits<double>::denorm_min();
  const Model model(ModelParameters{
      .particle_count = 1,
      .beta = beta,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const MatsubaraModeSet modes = make_continuous_modes(model.beta(), TorusLayout(3, 1), {{0}}, {0});
  const ContinuousConfiguration configuration(
      model, Permutation({0}),
      {ContinuousPath(beta, {0}, {3},
                      {{.time = 0.0, .axis = 0, .direction = 1},
                       {.time = 0.0, .axis = 0, .direction = 1},
                       {.time = 0.0, .axis = 0, .direction = 1}})});
  const ContinuousParticleModes values =
      continuous_particle_modes(configuration, ContinuousMatsubaraPlan(modes));
  HoppingResponseAccumulator accumulator(model, modes);

  accumulator.observe(values);
  EXPECT_EQ(accumulator.sample_count(), 1U);
  EXPECT_THROW(static_cast<void>(accumulator.finish()), std::overflow_error);
  EXPECT_EQ(accumulator.sample_count(), 1U);
}

} // namespace
} // namespace qmc
