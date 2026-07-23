#include "continuation_bundle.hpp"
#include "qmc/continuous_observables.hpp"
#include "qmc/interacting_sampler.hpp"
#include "qmc/path.hpp"
#include "qmc/version.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

class ScopedTemporaryDirectory {
public:
  ScopedTemporaryDirectory() {
    static std::atomic<std::uint64_t> sequence = 0;
    const auto timestamp =
        static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    for (std::uint64_t attempt = 0; attempt < 1024; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("qmc-continuation-statistical-" + std::to_string(timestamp) + "-" +
               std::to_string(sequence.fetch_add(1)) + "-" + std::to_string(attempt));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
      if (error && error != std::errc::file_exists) {
        throw std::filesystem::filesystem_error("failed to create statistical test directory",
                                                path_, error);
      }
    }
    throw std::runtime_error("failed to reserve a statistical test directory");
  }

  ~ScopedTemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  ScopedTemporaryDirectory(const ScopedTemporaryDirectory &) = delete;
  ScopedTemporaryDirectory &operator=(const ScopedTemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept { return path_; }

private:
  std::filesystem::path path_;
};

std::vector<std::string> split_tsv(const std::string &line) {
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (begin <= line.size()) {
    const std::size_t end = line.find('\t', begin);
    fields.push_back(
        line.substr(begin, end == std::string::npos ? line.size() - begin : end - begin));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return fields;
}

std::vector<std::vector<std::string>> read_tsv(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to read statistical continuation table: " + path.string());
  }
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(input, line)) {
    rows.push_back(split_tsv(line));
  }
  return rows;
}

std::string manifest_value(const std::vector<std::vector<std::string>> &rows,
                           const std::string &key) {
  for (std::size_t row = 1; row < rows.size(); ++row) {
    if (rows[row].size() == 2 && rows[row][0] == key) {
      return rows[row][1];
    }
  }
  throw std::runtime_error("missing continuation manifest key: " + key);
}

TEST(ContinuousBridgeDistributionTest, EventCountMatchesConditionedPoissonLaw) {
  constexpr double duration = 0.8;
  constexpr double hopping = 1.0;
  constexpr double lambda = duration * hopping;
  constexpr std::size_t sample_count = 30'000;

  double normalization = 0.0;
  double weighted_events = 0.0;
  double weight = 1.0;
  for (std::size_t pairs = 0; pairs < 100; ++pairs) {
    if (pairs > 0) {
      const auto count = static_cast<double>(pairs);
      weight *= (lambda * lambda) / (count * count);
    }
    normalization += weight;
    weighted_events += 2.0 * static_cast<double>(pairs) * weight;
  }
  const double exact_mean = weighted_events / normalization;

  qmc::Random random(8182);
  double event_sum = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const auto path = qmc::sample_continuous_bridge({0}, {0}, duration, hopping, random);
    event_sum += static_cast<double>(path.event_count());
  }
  EXPECT_NEAR(event_sum / static_cast<double>(sample_count), exact_mean, 0.035);
}

TEST(ContinuousMatsubaraAccumulatorTest, MatchesOneParticleLehmannReferences) {
  constexpr std::size_t sample_count = 40'000;
  const qmc::Model model(qmc::ModelParameters{
      .particle_count = 1,
      .beta = 0.8,
      .linear_size = 3,
      .dimension = 1,
      .hopping = 1.0,
  });
  const qmc::CanonicalEnsemble ensemble(model);
  const qmc::MatsubaraModeSet modes(
      model.beta(), qmc::TorusLayout(model.linear_size(), model.dimension()),
      qmc::MatsubaraModeRequest{.momentum_indices = {{0}, {1}}, .frequency_indices = {0, 1}});
  const qmc::ContinuousMatsubaraPlan plan(modes);
  const qmc::ImaginaryTimeLagSet lags(
      model.beta(), qmc::TorusLayout(model.linear_size(), model.dimension()),
      qmc::ImaginaryTimeLagRequest{.momentum_indices = {{0}, {1}, {2}}, .lags = {0.0, 0.2, 0.4}});
  const qmc::ContinuousDensityLagPlan lag_plan(lags);
  qmc::DensityMatsubaraAccumulator density_accumulator(model, modes);
  qmc::DensityLagBlockAccumulator lag_accumulator(model, lags, 1);
  qmc::HoppingResponseAccumulator hopping_accumulator(model, modes);
  qmc::Random random(20260722);
  std::array<double, 2> density_sums{};
  std::array<double, 2> density_square_sums{};
  std::array<double, 4> response_sums{};
  std::array<double, 4> response_square_sums{};
  double diamagnetic_sum = 0.0;
  double diamagnetic_square_sum = 0.0;

  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const qmc::ContinuousConfiguration configuration =
        qmc::sample_ideal_continuous_configuration(ensemble, random);
    const qmc::ContinuousMeasurementContext context(configuration);
    const qmc::ContinuousParticleModes values = qmc::continuous_particle_modes(context, plan);
    density_accumulator.observe(values);
    lag_accumulator.observe(qmc::continuous_density_lag_values(context, lag_plan));
    hopping_accumulator.observe(values);
    for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
      const double density_observation =
          std::norm(values.density(frequency, 1)) / (model.beta() * model.volume());
      density_sums[frequency] += density_observation;
      density_square_sums[frequency] += density_observation * density_observation;
      for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
        const std::size_t mode = (frequency * modes.momentum_count()) + momentum;
        const double response_observation =
            std::norm(values.flux(frequency, momentum, 0)) / (model.beta() * model.volume());
        response_sums[mode] += response_observation;
        response_square_sums[mode] += response_observation * response_observation;
      }
    }
    const double diamagnetic_observation =
        static_cast<double>(values.axis_event_count(0)) / (model.beta() * model.volume());
    diamagnetic_sum += diamagnetic_observation;
    diamagnetic_square_sum += diamagnetic_observation * diamagnetic_observation;
  }

  const qmc::ContinuousMatsubaraDensityCorrelations density_result = density_accumulator.finish();
  const qmc::DensityLagBlockSeries lag_result = lag_accumulator.finish();
  const qmc::HoppingResponse hopping_result = hopping_accumulator.finish();
  ASSERT_EQ(density_result.sample_count(), sample_count);
  ASSERT_EQ(lag_result.sample_count(), sample_count);
  ASSERT_EQ(hopping_result.sample_count(), sample_count);
  const std::array exact_density{0.1915077492, 0.0217766472};
  const double count = static_cast<double>(sample_count);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    const double sample_mean = density_sums[frequency] / count;
    const double sample_variance = (density_square_sums[frequency] -
                                    (density_sums[frequency] * density_sums[frequency] / count)) /
                                   (count - 1.0);
    const double standard_error = std::sqrt(sample_variance / count);
    EXPECT_NEAR(density_result.at(frequency, 1), sample_mean, 1e-13);
    // Independent one-particle Lehmann values at q=2*pi/3, with a six-standard-
    // error bound from exact independent ideal samples.
    EXPECT_NEAR(density_result.at(frequency, 1), exact_density[frequency], 6.0 * standard_error);
    EXPECT_LT(standard_error, frequency == 0 ? 0.0010 : 0.00035);
  }

  const std::array exact_lag_density{0.3333333333333333, 0.2270763548896555, 0.1955547971340791};
  for (std::size_t lag = 0; lag < lags.lag_count(); ++lag) {
    EXPECT_EQ(lag_result.mean(lag, 0), 0.0);
    EXPECT_EQ(lag_result.standard_error(lag, 0), 0.0);
    for (const std::size_t momentum : {std::size_t{1}, std::size_t{2}}) {
      if (lag == 0) {
        EXPECT_NEAR(lag_result.mean(lag, momentum), exact_lag_density[lag], 1e-13);
        EXPECT_NEAR(lag_result.standard_error(lag, momentum), 0.0, 1e-15);
      } else {
        // Independent one-particle Lehmann values at q=+/-2*pi/3. Each exact
        // configuration is one block, so this is the ordinary independent-sample
        // standard error without an autocorrelation assumption.
        EXPECT_NEAR(lag_result.mean(lag, momentum), exact_lag_density[lag],
                    6.0 * lag_result.standard_error(lag, momentum));
        EXPECT_LT(lag_result.standard_error(lag, momentum), 0.001);
      }
    }
  }

  // Independent one-particle source-derivative values. The mode order is
  // (n=0,q=0), (n=0,q=2*pi/3), (n=1,q=0), (n=1,q=2*pi/3).
  const std::array exact_response{0.3902364092, 0.0, 0.5130943014, 0.4477643599};
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      const std::size_t mode = (frequency * modes.momentum_count()) + momentum;
      const double sample_mean = response_sums[mode] / count;
      const double sample_variance =
          (response_square_sums[mode] - (response_sums[mode] * response_sums[mode] / count)) /
          (count - 1.0);
      const double standard_error = std::sqrt(std::max(0.0, sample_variance) / count);
      EXPECT_NEAR(hopping_result.flux_response(frequency, momentum, 0, 0).real(), sample_mean,
                  1e-13);
      EXPECT_EQ(hopping_result.flux_response(frequency, momentum, 0, 0).imag(), 0.0);
      if (exact_response[mode] == 0.0) {
        EXPECT_NEAR(hopping_result.flux_response(frequency, momentum, 0, 0).real(), 0.0, 1e-24);
      } else {
        EXPECT_NEAR(hopping_result.flux_response(frequency, momentum, 0, 0).real(),
                    exact_response[mode], 6.0 * standard_error);
        EXPECT_LT(standard_error, 0.01);
      }
    }
  }

  const double diamagnetic_mean = diamagnetic_sum / count;
  const double diamagnetic_variance =
      (diamagnetic_square_sum - (diamagnetic_sum * diamagnetic_sum / count)) / (count - 1.0);
  const double diamagnetic_standard_error = std::sqrt(diamagnetic_variance / count);
  EXPECT_NEAR(hopping_result.diamagnetic(0), diamagnetic_mean, 1e-13);
  EXPECT_NEAR(hopping_result.diamagnetic(0), 0.5130943014, 6.0 * diamagnetic_standard_error);
  EXPECT_LT(diamagnetic_standard_error, 0.005);
}

TEST(InteractingDistributionTest, MatchesSmallSystemExactDiagonalizationReference) {
  constexpr std::uint64_t seed = 20260717;
  constexpr std::size_t burn_in = 2'000;
  constexpr std::size_t sample_count = 30'000;
  constexpr std::size_t block_size = 100;
  constexpr std::size_t block_count = sample_count / block_size;
  static_assert(sample_count % block_size == 0);
  const qmc::InteractingModel model{
      .free = qmc::Model(qmc::ModelParameters{
          .particle_count = 2, .beta = 0.8, .linear_size = 3, .dimension = 1, .hopping = 1.0}),
      .interaction = 1.2,
  };
  const qmc::MatsubaraModeSet modes(
      model.free.beta(), qmc::TorusLayout(model.free.linear_size(), model.free.dimension()),
      qmc::MatsubaraModeRequest{.momentum_indices = {{1}}, .frequency_indices = {0, 1}});
  const qmc::ContinuousMatsubaraPlan plan(modes);
  qmc::DensityMatsubaraAccumulator density_accumulator(model.free, modes);
  qmc::DensityMatsubaraBlockAccumulator density_block_accumulator(model.free, modes, block_size);
  qmc::HoppingResponseAccumulator hopping_accumulator(model.free, modes);
  qmc::HoppingResponseBlockAccumulator hopping_block_accumulator(model.free, modes, block_size);
  const qmc::SweepOptions sweep{
      .segment_updates = 0,
      .cycle_updates = 0,
      .global_updates = 1,
      .stitch_updates = 0,
      .time_shift_updates = 0,
  };
  const qmc::RandomSeamStitchOptions random_seam_stitch{.updates = 0};
  qmc::InteractingSampler sampler(model, seed);
  const auto advance = [&sampler, &sweep, &random_seam_stitch] {
    sampler.random_seam_stitch_sweep(random_seam_stitch);
    sampler.sweep(sweep);
  };
  for (std::size_t update = 0; update < burn_in; ++update) {
    advance();
  }

  double total_energy = 0.0;
  double kinetic_energy = 0.0;
  double interaction_energy = 0.0;
  double pair_count = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    advance();
    const auto value = sampler.observables();
    total_energy += value.total_energy;
    kinetic_energy += value.kinetic_energy;
    interaction_energy += value.interaction_energy;
    pair_count += value.pair_overlap_time / model.free.beta();

    const qmc::ContinuousParticleModes modes_sample =
        qmc::continuous_particle_modes(sampler.state(), plan);
    density_accumulator.observe(modes_sample);
    density_block_accumulator.observe(modes_sample);
    hopping_accumulator.observe(modes_sample);
    hopping_block_accumulator.observe(modes_sample);
  }

  const double denominator = static_cast<double>(sample_count);
  // Independent Fock-space diagonalization values from python/validate_interacting_ed.py.
  EXPECT_NEAR(total_energy / denominator, -3.1421047841, 0.07);
  EXPECT_NEAR(kinetic_energy / denominator, -3.4840757439, 0.07);
  EXPECT_NEAR(interaction_energy / denominator, 0.3419709598, 0.018);
  EXPECT_NEAR(pair_count / denominator, 0.2849757998, 0.015);

  const qmc::ContinuousMatsubaraDensityCorrelations density_result = density_accumulator.finish();
  const qmc::DensityMatsubaraBlockSeries density_series = density_block_accumulator.finish();
  const qmc::HoppingResponse hopping_result = hopping_accumulator.finish();
  const qmc::HoppingResponseBlockSeries hopping_series = hopping_block_accumulator.finish();
  ASSERT_EQ(density_result.sample_count(), sample_count);
  ASSERT_EQ(density_series.model(), model.free);
  ASSERT_EQ(density_series.modes(), modes);
  ASSERT_EQ(density_series.measurements_per_block(), block_size);
  ASSERT_EQ(density_series.block_count(), block_count);
  ASSERT_EQ(density_series.sample_count(), sample_count);
  ASSERT_EQ(hopping_result.sample_count(), sample_count);
  ASSERT_EQ(hopping_series.model(), model.free);
  ASSERT_EQ(hopping_series.modes(), modes);
  ASSERT_EQ(hopping_series.measurements_per_block(), block_size);
  ASSERT_EQ(hopping_series.block_count(), block_count);
  ASSERT_EQ(hopping_series.sample_count(), sample_count);
  const std::array exact_density{0.32798, 0.04773};
  for (std::size_t frequency = 0; frequency < exact_density.size(); ++frequency) {
    EXPECT_NEAR(density_result.at(frequency, 0), density_series.mean(frequency, 0), 1e-13);
    EXPECT_NEAR(density_series.mean(frequency, 0), exact_density[frequency],
                6.0 * density_series.standard_error(frequency, 0))
        << "blocked standard error = " << density_series.standard_error(frequency, 0);
  }
  const double covariance_00 = density_series.covariance_of_mean(0, 0, 0);
  const double covariance_01 = density_series.covariance_of_mean(0, 0, 1);
  const double covariance_10 = density_series.covariance_of_mean(0, 1, 0);
  const double covariance_11 = density_series.covariance_of_mean(0, 1, 1);
  EXPECT_TRUE(std::isfinite(covariance_00));
  EXPECT_TRUE(std::isfinite(covariance_01));
  EXPECT_TRUE(std::isfinite(covariance_10));
  EXPECT_TRUE(std::isfinite(covariance_11));
  EXPECT_EQ(covariance_01, covariance_10);
  EXPECT_NEAR(covariance_00, std::pow(density_series.standard_error(0, 0), 2), 1e-18);
  EXPECT_NEAR(covariance_11, std::pow(density_series.standard_error(1, 0), 2), 1e-18);
  EXPECT_GE((covariance_00 * covariance_11) - (covariance_01 * covariance_10), -1e-18);

  EXPECT_NEAR(hopping_result.flux_response(1, 0, 0, 0).real(),
              hopping_series.flux_response(1, 0, 0, 0).real(), 1e-13);
  EXPECT_EQ(hopping_result.flux_response(1, 0, 0, 0).imag(), 0.0);
  const double response_standard_error =
      hopping_series.flux_response_standard_error(1, 0, 0, 0, qmc::HoppingResponseComponent::Real);
  EXPECT_NEAR(hopping_result.flux_response(1, 0, 0, 0).real(), 0.98151,
              6.0 * response_standard_error)
      << "blocked standard error = " << response_standard_error;

  EXPECT_NEAR(hopping_result.diamagnetic(0), hopping_series.diamagnetic(0), 1e-13);
  const double diamagnetic_standard_error = hopping_series.diamagnetic_standard_error(0);
  EXPECT_NEAR(hopping_result.diamagnetic(0), 1.16136, 6.0 * diamagnetic_standard_error)
      << "blocked standard error = " << diamagnetic_standard_error;

  // These caps keep the six-standard-error comparisons discriminating and
  // guard against a silently under-mixed chain.
  EXPECT_LT(density_series.standard_error(0, 0), 0.004);
  EXPECT_LT(density_series.standard_error(1, 0), 0.001);
  EXPECT_LT(response_standard_error, 0.02);
  EXPECT_LT(diamagnetic_standard_error, 0.01);

  ScopedTemporaryDirectory temporary;
  const std::filesystem::path bundle = temporary.path() / "density-continuation-v1";
  qmc::example::write_density_continuation_bundle(bundle, density_series,
                                                  {
                                                      .model = model,
                                                      .seed = seed,
                                                      .burn_in_sweeps = burn_in,
                                                      .thinning_sweeps = 1,
                                                      .sweep = sweep,
                                                      .random_seam_stitch = random_seam_stitch,
                                                      .scalar_trace_retained = false,
                                                      .program = "qmc_statistical_tests",
                                                      .program_version = std::string(qmc::kVersion),
                                                  });

  const auto manifest_rows = read_tsv(bundle / "manifest.tsv");
  ASSERT_FALSE(manifest_rows.empty());
  EXPECT_EQ(manifest_rows.front(), std::vector<std::string>({"key", "value"}));
  EXPECT_EQ(manifest_value(manifest_rows, "model_interaction"), "1.2");
  EXPECT_EQ(manifest_value(manifest_rows, "seed"), std::to_string(seed));
  EXPECT_EQ(manifest_value(manifest_rows, "burn_in_sweeps"), std::to_string(burn_in));
  EXPECT_EQ(manifest_value(manifest_rows, "thinning_sweeps"), "1");
  EXPECT_EQ(manifest_value(manifest_rows, "sweep_global_updates"), "1");
  EXPECT_EQ(manifest_value(manifest_rows, "random_seam_stitch_updates"), "0");
  EXPECT_EQ(manifest_value(manifest_rows, "measurements_per_block"), std::to_string(block_size));
  EXPECT_EQ(manifest_value(manifest_rows, "completed_block_count"), std::to_string(block_count));
  EXPECT_EQ(manifest_value(manifest_rows, "sample_count"), std::to_string(sample_count));
  EXPECT_EQ(manifest_value(manifest_rows, "covariance_rank_status"), "full_rank_possible");

  const auto value_rows = read_tsv(bundle / "values.tsv");
  ASSERT_EQ(value_rows.size(), modes.frequency_count() + 1);
  EXPECT_EQ(value_rows.front(),
            std::vector<std::string>({"momentum", "k_0", "frequency", "n", "omega", "mean",
                                      "standard_error", "exact_constraint"}));
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    const auto &row = value_rows[frequency + 1];
    ASSERT_EQ(row.size(), 8U);
    EXPECT_EQ(std::stoull(row[0]), 0U);
    EXPECT_EQ(std::stoull(row[1]), 1U);
    EXPECT_EQ(std::stoull(row[2]), frequency);
    EXPECT_EQ(std::stoll(row[3]), modes.frequency_index(frequency));
    EXPECT_DOUBLE_EQ(std::stod(row[4]), modes.frequency(frequency));
    EXPECT_DOUBLE_EQ(std::stod(row[5]), density_series.mean(frequency, 0));
    EXPECT_DOUBLE_EQ(std::stod(row[6]), density_series.standard_error(frequency, 0));
    EXPECT_EQ(row[7], "0");
  }

  const auto covariance_rows = read_tsv(bundle / "covariance.tsv");
  ASSERT_EQ(covariance_rows.size(), (modes.frequency_count() * modes.frequency_count()) + 1);
  EXPECT_EQ(covariance_rows.front(),
            std::vector<std::string>({"momentum", "left", "right", "covariance_of_mean"}));
  for (std::size_t row_index = 1; row_index < covariance_rows.size(); ++row_index) {
    const auto &row = covariance_rows[row_index];
    ASSERT_EQ(row.size(), 4U);
    const std::size_t left = std::stoull(row[1]);
    const std::size_t right = std::stoull(row[2]);
    EXPECT_EQ(std::stoull(row[0]), 0U);
    ASSERT_LT(left, modes.frequency_count());
    ASSERT_LT(right, modes.frequency_count());
    EXPECT_DOUBLE_EQ(std::stod(row[3]), density_series.covariance_of_mean(0, left, right));
  }

  const auto block_rows = read_tsv(bundle / "blocks.tsv");
  ASSERT_EQ(block_rows.size(), (block_count * modes.frequency_count()) + 1);
  EXPECT_EQ(block_rows.front(),
            std::vector<std::string>({"block", "momentum", "frequency_or_lag", "value"}));
  std::vector<std::array<double, 2>> exported_blocks(block_count);
  std::vector<bool> exported_block_seen(block_count * modes.frequency_count(), false);
  for (std::size_t row_index = 1; row_index < block_rows.size(); ++row_index) {
    const auto &row = block_rows[row_index];
    ASSERT_EQ(row.size(), 4U);
    const std::size_t block = std::stoull(row[0]);
    const std::size_t frequency = std::stoull(row[2]);
    ASSERT_LT(block, block_count);
    ASSERT_LT(frequency, modes.frequency_count());
    EXPECT_EQ(std::stoull(row[1]), 0U);
    const std::size_t seen_index = (block * modes.frequency_count()) + frequency;
    EXPECT_FALSE(exported_block_seen[seen_index]);
    exported_block_seen[seen_index] = true;
    exported_blocks[block][frequency] = std::stod(row[3]);
    EXPECT_DOUBLE_EQ(exported_blocks[block][frequency],
                     density_series.block_value(block, frequency, 0));
  }
  EXPECT_TRUE(std::ranges::all_of(exported_block_seen, [](const bool seen) { return seen; }));

  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    double exported_mean = 0.0;
    for (std::size_t block = 0; block < block_count; ++block) {
      exported_mean += exported_blocks[block][frequency] / static_cast<double>(block_count);
    }
    EXPECT_NEAR(exported_mean, density_series.mean(frequency, 0), 1e-14);

    for (const std::size_t omitted_block : {std::size_t{0}, block_count - 1}) {
      double exported_jackknife_mean = 0.0;
      for (std::size_t block = 0; block < block_count; ++block) {
        if (block != omitted_block) {
          exported_jackknife_mean +=
              exported_blocks[block][frequency] / static_cast<double>(block_count - 1);
        }
      }
      EXPECT_NEAR(exported_jackknife_mean,
                  density_series.jackknife_mean(omitted_block, frequency, 0), 1e-14);
    }
  }
  for (std::size_t left = 0; left < modes.frequency_count(); ++left) {
    for (std::size_t right = 0; right < modes.frequency_count(); ++right) {
      double exported_covariance = 0.0;
      for (std::size_t block = 0; block < block_count; ++block) {
        exported_covariance += (exported_blocks[block][left] - density_series.mean(left, 0)) *
                               (exported_blocks[block][right] - density_series.mean(right, 0));
      }
      exported_covariance /= static_cast<double>(block_count * (block_count - 1));
      EXPECT_NEAR(exported_covariance, density_series.covariance_of_mean(0, left, right), 1e-16);
    }
  }

  const std::filesystem::path hopping_bundle = temporary.path() / "hopping-response-v1";
  qmc::example::write_hopping_response_bundle(hopping_bundle, hopping_series,
                                              {
                                                  .model = model,
                                                  .seed = seed,
                                                  .burn_in_sweeps = burn_in,
                                                  .thinning_sweeps = 1,
                                                  .sweep = sweep,
                                                  .random_seam_stitch = random_seam_stitch,
                                                  .scalar_trace_retained = false,
                                                  .program = "qmc_statistical_tests",
                                                  .program_version = std::string(qmc::kVersion),
                                              });
  const auto hopping_manifest_rows = read_tsv(hopping_bundle / "manifest.tsv");
  EXPECT_EQ(manifest_value(hopping_manifest_rows, "schema_id"), "hopping-response");
  EXPECT_EQ(manifest_value(hopping_manifest_rows, "model_interaction"), "1.2");
  EXPECT_EQ(manifest_value(hopping_manifest_rows, "completed_block_count"),
            std::to_string(block_count));
  const auto hopping_response_rows = read_tsv(hopping_bundle / "response_values.tsv");
  ASSERT_EQ(hopping_response_rows.size(), modes.frequency_count() + 1);
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    const auto &row = hopping_response_rows[frequency + 1];
    ASSERT_EQ(row.size(), 15U);
    EXPECT_DOUBLE_EQ(std::stod(row[7]), hopping_series.flux_response(frequency, 0, 0, 0).real());
    EXPECT_DOUBLE_EQ(std::stod(row[9]),
                     hopping_series.flux_response_standard_error(
                         frequency, 0, 0, 0, qmc::HoppingResponseComponent::Real));
  }

  const auto acceptance = sampler.statistics(qmc::MoveKind::GlobalMove).acceptance();
  ASSERT_TRUE(acceptance.has_value());
  EXPECT_GT(*acceptance, 0.75);
  EXPECT_LT(*acceptance, 0.90);
}

TEST(InteractingDistributionTest, StrongCouplingStitchKernelMatchesExactDiagonalization) {
  constexpr std::size_t burn_in = 5'000;
  constexpr std::size_t sample_count = 30'000;
  constexpr double beta = 2.0;
  constexpr double interaction = 16.0;
  const qmc::InteractingModel model{
      .free = qmc::Model(qmc::ModelParameters{
          .particle_count = 4,
          .beta = beta,
          .linear_size = 4,
          .dimension = 1,
          .hopping = 1.0,
      }),
      .interaction = interaction,
  };
  qmc::InteractingSampler sampler(model, 10101);
  const qmc::RandomSeamStitchOptions stitches{
      .updates = model.free.particle_count(),
      // Strong-coupling calibration: redraw a slab of duration 5/U.
      .fraction = 5.0 / (interaction * beta),
  };
  const qmc::SweepOptions local_geometry{
      .segment_updates = model.free.particle_count(),
      .segment_fraction = 0.35,
      .cycle_updates = 1,
      .global_updates = 0,
  };
  const auto advance = [&sampler, &stitches, &local_geometry] {
    sampler.random_seam_stitch_sweep(stitches);
    sampler.sweep(local_geometry);
  };

  for (std::size_t update = 0; update < burn_in; ++update) {
    advance();
  }

  double total_energy = 0.0;
  double kinetic_energy = 0.0;
  double interaction_energy = 0.0;
  double double_occupancy = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    advance();
    const auto value = sampler.observables();
    total_energy += value.total_energy;
    kinetic_energy += value.kinetic_energy;
    interaction_energy += value.interaction_energy;
    double_occupancy += value.double_occupancy_per_site;
  }

  const double denominator = static_cast<double>(sample_count);
  // Independent Fock-space diagonalization values from python/validate_interacting_ed.py.
  EXPECT_NEAR(total_energy / denominator, -1.0339175138, 0.075);
  EXPECT_NEAR(kinetic_energy / denominator, -2.1253604095, 0.14);
  EXPECT_NEAR(interaction_energy / denominator, 1.0914428957, 0.09);
  EXPECT_NEAR(double_occupancy / denominator, 0.0170537952, 0.0015);

  const qmc::MoveStatistics &stitch = sampler.statistics(qmc::MoveKind::StitchMove);
  const auto stitch_acceptance = stitch.acceptance();
  const auto topology_rate = stitch.topology_change_rate();
  ASSERT_TRUE(stitch_acceptance.has_value());
  ASSERT_TRUE(topology_rate.has_value());
  EXPECT_GT(*stitch_acceptance, 0.6);
  EXPECT_LT(*stitch_acceptance, 0.95);
  EXPECT_GT(*topology_rate, 0.03);
  EXPECT_EQ(sampler.statistics(qmc::MoveKind::GlobalMove).attempts, 0U);
}

} // namespace
