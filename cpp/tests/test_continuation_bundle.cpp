#include "continuation_bundle.hpp"
#include "qmc/interacting.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qmc::example {
namespace {

class ScopedTemporaryDirectory {
public:
  ScopedTemporaryDirectory() {
    static std::atomic<std::uint64_t> sequence = 0;
    const auto timestamp =
        static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    for (std::uint64_t attempt = 0; attempt < 1024; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("qmc-continuation-test-" + std::to_string(timestamp) + "-" +
               std::to_string(sequence.fetch_add(1)) + "-" + std::to_string(attempt));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
      if (error && error != std::errc::file_exists) {
        throw std::filesystem::filesystem_error("failed to create test directory", path_, error);
      }
    }
    throw std::runtime_error("failed to reserve a unique continuation test directory");
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

InteractingModel test_model() {
  return {
      .free = Model(ModelParameters{
          .particle_count = 1,
          .beta = 1.0,
          .linear_size = 2,
          .dimension = 1,
          .hopping = 1.0,
      }),
      .interaction = 0.75,
  };
}

MatsubaraModeSet test_modes(const Model &model,
                            std::vector<std::vector<std::size_t>> momenta = {{0}, {1}},
                            std::vector<std::int64_t> frequencies = {0, 1, 2}) {
  return {model.beta(),
          TorusLayout(model.linear_size(), model.dimension()),
          {.momentum_indices = std::move(momenta), .frequency_indices = std::move(frequencies)}};
}

DensityMatsubaraBlockSeries make_series(const Model &model, const MatsubaraModeSet &modes) {
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
  DensityMatsubaraBlockAccumulator accumulator(model, modes, 2);
  for (const ContinuousParticleModes *values : {&static_values, &static_values, &static_values,
                                                &moving_values, &moving_values, &moving_values}) {
    accumulator.observe(*values);
  }
  return accumulator.finish();
}

DensityContinuationRunProvenance test_provenance(const InteractingModel &model) {
  return {
      .model = model,
      .seed = 123456,
      .burn_in_sweeps = 11,
      .thinning_sweeps = 3,
      .sweep =
          {
              .segment_updates = 4,
              .segment_fraction = 0.2,
              .cycle_updates = 2,
              .global_updates = 1,
              .stitch_updates = 0,
              .stitch_fraction = 0.4,
              .stitch_locality_radius = 2,
              .stitch_global_partner_probability = 0.1,
              .stitch_mixture = {.strand_counts = {2, 3}, .strand_weights = {0.8, 0.2}},
              .time_shift_updates = 1,
          },
      .random_seam_stitch =
          {
              .updates = 5,
              .fraction = 0.6,
              .locality_radius = 3,
              .global_partner_probability = 0.15,
              .mixture = {.strand_counts = {2, 3}, .strand_weights = {0.7, 0.3}},
          },
      .scalar_trace_retained = false,
      .program = "qmc_test",
      .program_version = "9.8.7",
  };
}

std::vector<std::string> split(const std::string &line) {
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

std::vector<std::vector<std::string>> read_table(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to read table: " + path.string());
  }
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(input, line)) {
    rows.push_back(split(line));
  }
  return rows;
}

std::map<std::string, std::string> read_manifest(const std::filesystem::path &path) {
  const auto rows = read_table(path);
  if (rows.empty() || rows.front() != std::vector<std::string>({"key", "value"})) {
    throw std::runtime_error("manifest header is invalid");
  }
  std::map<std::string, std::string> manifest;
  for (std::size_t row = 1; row < rows.size(); ++row) {
    if (rows[row].size() != 2 || !manifest.emplace(rows[row][0], rows[row][1]).second) {
      throw std::runtime_error("manifest row is invalid");
    }
  }
  return manifest;
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

TEST(DensityContinuationBundleTest, WritesReproducibleTablesAndCompleteProvenance) {
  const InteractingModel model = test_model();
  const MatsubaraModeSet modes = test_modes(model.free);
  const DensityMatsubaraBlockSeries series = make_series(model.free, modes);
  const DensityContinuationRunProvenance provenance = test_provenance(model);
  ScopedTemporaryDirectory temporary;
  const std::filesystem::path destination = temporary.path() / "density-continuation-v1";

  validate_density_continuation_bundle_destination(destination);
  write_density_continuation_bundle(destination, series, provenance);

  for (const std::string filename :
       {"manifest.tsv", "values.tsv", "covariance.tsv", "blocks.tsv"}) {
    EXPECT_TRUE(std::filesystem::is_regular_file(destination / filename));
    EXPECT_GT(std::filesystem::file_size(destination / filename), 0U);
  }

  const auto manifest = read_manifest(destination / "manifest.tsv");
  EXPECT_EQ(manifest.at("schema_id"), "density-continuation");
  EXPECT_EQ(manifest.at("schema_version"), "1");
  EXPECT_EQ(manifest.at("basis"), "bosonic_matsubara");
  EXPECT_EQ(manifest.at("observable_id"), "connected_density_per_site");
  EXPECT_EQ(manifest.at("model_particle_count"), "1");
  EXPECT_EQ(manifest.at("model_interaction"), "0.75");
  EXPECT_EQ(manifest.at("seed"), "123456");
  EXPECT_EQ(manifest.at("burn_in_sweeps"), "11");
  EXPECT_EQ(manifest.at("thinning_sweeps"), "3");
  EXPECT_EQ(manifest.at("measurements_per_block"), "2");
  EXPECT_EQ(manifest.at("completed_block_count"), "3");
  EXPECT_EQ(manifest.at("sample_count"), "6");
  EXPECT_EQ(manifest.at("post_burn_in_sampler_sweeps_per_block"), "6");
  EXPECT_EQ(manifest.at("values_row_count"), "6");
  EXPECT_EQ(manifest.at("covariance_row_count"), "18");
  EXPECT_EQ(manifest.at("blocks_row_count"), "18");
  EXPECT_EQ(manifest.at("row_roles_present"), "measured,exact_constraint");
  EXPECT_EQ(manifest.at("covariance_rank_upper_bound"), "2");
  EXPECT_EQ(manifest.at("covariance_full_rank_possible"), "false");
  EXPECT_EQ(manifest.at("covariance_rank_status"), "rank_deficient_by_completed_block_count");
  EXPECT_EQ(manifest.at("covariance_regularization"), "none");
  EXPECT_EQ(manifest.at("sweep_stitch_strand_counts"), "2,3");
  EXPECT_EQ(manifest.at("random_seam_stitch_strand_weights"),
            "0.69999999999999996,0.29999999999999999");
  EXPECT_EQ(manifest.at("scalar_trace_retained"), "false");
  EXPECT_EQ(manifest.at("program"), "qmc_test");
  EXPECT_EQ(manifest.at("program_version"), "9.8.7");

  const auto block_rows = read_table(destination / "blocks.tsv");
  ASSERT_EQ(block_rows.size(), 19U);
  EXPECT_EQ(block_rows.front(),
            std::vector<std::string>({"block", "momentum", "frequency_or_lag", "value"}));
  std::map<std::tuple<std::size_t, std::size_t, std::size_t>, double> blocks;
  for (std::size_t row = 1; row < block_rows.size(); ++row) {
    ASSERT_EQ(block_rows[row].size(), 4U);
    blocks.emplace(std::make_tuple(std::stoull(block_rows[row][0]), std::stoull(block_rows[row][1]),
                                   std::stoull(block_rows[row][2])),
                   std::stod(block_rows[row][3]));
  }
  for (std::size_t omitted = 0; omitted < series.block_count(); ++omitted) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
        double jackknife = 0.0;
        for (std::size_t block = 0; block < series.block_count(); ++block) {
          if (block != omitted) {
            jackknife += blocks.at({block, momentum, frequency}) /
                         static_cast<double>(series.block_count() - 1);
          }
        }
        EXPECT_DOUBLE_EQ(series.jackknife_mean(omitted, frequency, momentum), jackknife);
      }
    }
  }

  const auto value_rows = read_table(destination / "values.tsv");
  ASSERT_EQ(value_rows.size(), 7U);
  EXPECT_EQ(value_rows.front(),
            std::vector<std::string>({"momentum", "k_0", "frequency", "n", "omega", "mean",
                                      "standard_error", "exact_constraint"}));
  for (std::size_t row = 1; row < value_rows.size(); ++row) {
    ASSERT_EQ(value_rows[row].size(), 8U);
    const std::size_t momentum = std::stoull(value_rows[row][0]);
    const std::size_t frequency = std::stoull(value_rows[row][2]);
    double mean = 0.0;
    for (std::size_t block = 0; block < series.block_count(); ++block) {
      mean += blocks.at({block, momentum, frequency}) / static_cast<double>(series.block_count());
    }
    EXPECT_DOUBLE_EQ(std::stod(value_rows[row][5]), mean);
    EXPECT_EQ(value_rows[row][7], momentum == 0 ? "1" : "0");
    if (momentum == 0) {
      EXPECT_EQ(std::stod(value_rows[row][5]), 0.0);
      EXPECT_EQ(std::stod(value_rows[row][6]), 0.0);
    }
  }

  const auto covariance_rows = read_table(destination / "covariance.tsv");
  ASSERT_EQ(covariance_rows.size(), 19U);
  EXPECT_EQ(covariance_rows.front(),
            std::vector<std::string>({"momentum", "left", "right", "covariance_of_mean"}));
  for (std::size_t row = 1; row < covariance_rows.size(); ++row) {
    ASSERT_EQ(covariance_rows[row].size(), 4U);
    const std::size_t momentum = std::stoull(covariance_rows[row][0]);
    const std::size_t left = std::stoull(covariance_rows[row][1]);
    const std::size_t right = std::stoull(covariance_rows[row][2]);
    double left_mean = 0.0;
    double right_mean = 0.0;
    for (std::size_t block = 0; block < series.block_count(); ++block) {
      left_mean += blocks.at({block, momentum, left}) / static_cast<double>(series.block_count());
      right_mean += blocks.at({block, momentum, right}) / static_cast<double>(series.block_count());
    }
    double covariance = 0.0;
    for (std::size_t block = 0; block < series.block_count(); ++block) {
      covariance += (blocks.at({block, momentum, left}) - left_mean) *
                    (blocks.at({block, momentum, right}) - right_mean);
    }
    covariance /= static_cast<double>(series.block_count() * (series.block_count() - 1));
    EXPECT_NEAR(std::stod(covariance_rows[row][3]), covariance, 1e-17);
  }
}

TEST(DensityContinuationBundleTest, RejectsExistingDestinationWithoutOverwritingIt) {
  const InteractingModel model = test_model();
  const MatsubaraModeSet modes = test_modes(model.free);
  const DensityMatsubaraBlockSeries series = make_series(model.free, modes);
  ScopedTemporaryDirectory temporary;
  const std::filesystem::path destination = temporary.path() / "bundle";
  write_density_continuation_bundle(destination, series, test_provenance(model));
  const std::string original_manifest = read_file(destination / "manifest.tsv");

  EXPECT_THROW(validate_density_continuation_bundle_destination(destination),
               std::invalid_argument);
  EXPECT_THROW(write_density_continuation_bundle(destination, series, test_provenance(model)),
               std::invalid_argument);
  EXPECT_EQ(read_file(destination / "manifest.tsv"), original_manifest);
}

TEST(DensityContinuationBundleTest, RejectsInvalidRowsBeforeCreatingTemporaryOutput) {
  const InteractingModel model = test_model();
  const MatsubaraModeSet zero_modes = test_modes(model.free, {{0}}, {0, 1});
  const DensityMatsubaraBlockSeries zero_series = make_series(model.free, zero_modes);
  ScopedTemporaryDirectory temporary;
  const std::filesystem::path destination = temporary.path() / "bundle";

  EXPECT_THROW(write_density_continuation_bundle(destination, zero_series, test_provenance(model)),
               std::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(destination));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(temporary.path()),
                          std::filesystem::directory_iterator()),
            0);

  const MatsubaraModeSet negative_modes = test_modes(model.free, {{1}}, {-1, 0});
  const DensityMatsubaraBlockSeries negative_series = make_series(model.free, negative_modes);
  EXPECT_THROW(
      write_density_continuation_bundle(destination, negative_series, test_provenance(model)),
      std::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(destination));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(temporary.path()),
                          std::filesystem::directory_iterator()),
            0);
}

} // namespace
} // namespace qmc::example
