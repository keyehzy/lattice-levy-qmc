#include "continuation_bundle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace qmc::example {
namespace {

constexpr std::string_view kSchemaIdentifier = "density-continuation";
constexpr std::string_view kSchemaVersion = "1";

struct BundleBasisMetadata {
  std::string_view basis;
  std::string_view temporal_phase;
  std::string_view normalization;
  std::string_view value_units;
  std::string_view point_units_key;
  std::string_view point_units;
  std::string_view point_count_key;
  std::string_view covariance_scope;
  bool values_may_be_negative;
};

BundleBasisMetadata bundle_metadata([[maybe_unused]] const DensityMatsubaraBlockSeries &series) {
  return {
      .basis = "bosonic_matsubara",
      .temporal_phase = "exp(+i*omega_n*tau)",
      .normalization = "mean_abs_centered_density_amplitude_squared_over_beta_volume",
      .value_units = "inverse_energy_per_site",
      .point_units_key = "frequency_units",
      .point_units = "energy",
      .point_count_key = "frequency_count",
      .covariance_scope = "independent_dense_frequency_matrix_per_momentum",
      .values_may_be_negative = false,
  };
}

BundleBasisMetadata bundle_metadata([[maybe_unused]] const DensityLagBlockSeries &series) {
  return {
      .basis = "imaginary_time_lag",
      .temporal_phase = "not_applicable",
      .normalization = "mean_connected_density_time_overlap_over_beta_volume",
      .value_units = "dimensionless_per_site",
      .point_units_key = "lag_units",
      .point_units = "inverse_energy",
      .point_count_key = "lag_count",
      .covariance_scope = "independent_dense_lag_matrix_per_momentum",
      .values_may_be_negative = true,
  };
}

const MatsubaraModeSet &series_geometry(const DensityMatsubaraBlockSeries &series) {
  return series.modes();
}

const ImaginaryTimeLagSet &series_geometry(const DensityLagBlockSeries &series) {
  return series.lags();
}

std::size_t series_point_count(const DensityMatsubaraBlockSeries &series) {
  return series.modes().frequency_count();
}

std::size_t series_point_count(const DensityLagBlockSeries &series) {
  return series.lags().lag_count();
}

double series_mean(const DensityMatsubaraBlockSeries &series, const std::size_t point,
                   const std::size_t momentum) {
  return series.mean(point, momentum);
}

double series_mean(const DensityLagBlockSeries &series, const std::size_t point,
                   const std::size_t momentum) {
  return series.mean(point, momentum);
}

double series_standard_error(const DensityMatsubaraBlockSeries &series, const std::size_t point,
                             const std::size_t momentum) {
  return series.standard_error(point, momentum);
}

double series_standard_error(const DensityLagBlockSeries &series, const std::size_t point,
                             const std::size_t momentum) {
  return series.standard_error(point, momentum);
}

double series_covariance(const DensityMatsubaraBlockSeries &series, const std::size_t momentum,
                         const std::size_t left, const std::size_t right) {
  return series.covariance_of_mean(momentum, left, right);
}

double series_covariance(const DensityLagBlockSeries &series, const std::size_t momentum,
                         const std::size_t left, const std::size_t right) {
  return series.covariance_of_mean(momentum, left, right);
}

double series_block_value(const DensityMatsubaraBlockSeries &series, const std::size_t block,
                          const std::size_t point, const std::size_t momentum) {
  return series.block_value(block, point, momentum);
}

double series_block_value(const DensityLagBlockSeries &series, const std::size_t block,
                          const std::size_t point, const std::size_t momentum) {
  return series.block_value(block, point, momentum);
}

std::size_t checked_product(const std::size_t left, const std::size_t right,
                            const char *description) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(description);
  }
  return left * right;
}

void validate_tsv_atom(const std::string_view value, const char *description) {
  if (value.empty()) {
    throw std::invalid_argument(std::string(description) + " must not be empty");
  }
  if (value.find_first_of("\t\r\n") != std::string_view::npos) {
    throw std::invalid_argument(std::string(description) + " contains a TSV delimiter");
  }
}

template <class Geometry>
bool is_zero_momentum(const Geometry &geometry, const std::size_t momentum) {
  return std::ranges::all_of(geometry.momentum_indices(momentum),
                             [](const std::size_t component) { return component == 0; });
}

void validate_probability(const double value, const char *description) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    throw std::invalid_argument(std::string(description) + " must lie in [0, 1]");
  }
}

void validate_fraction(const double value, const char *description) {
  if (!std::isfinite(value) || value <= 0.0 || value > 1.0) {
    throw std::invalid_argument(std::string(description) + " must lie in (0, 1]");
  }
}

void validate_stitch_mixture(const StitchMixture &mixture, const std::size_t particle_count,
                             const char *description) {
  if (mixture.strand_counts.empty()) {
    throw std::invalid_argument(std::string(description) + " requires at least one strand count");
  }
  std::vector<std::size_t> counts;
  counts.reserve(mixture.strand_counts.size());
  for (const std::size_t count : mixture.strand_counts) {
    if (count < 2 || count > 8) {
      throw std::invalid_argument(std::string(description) + " strand counts must lie in [2, 8]");
    }
    if (std::ranges::find(counts, count) != counts.end()) {
      throw std::invalid_argument(std::string(description) + " strand counts must be unique");
    }
    counts.push_back(count);
  }
  if (!mixture.strand_weights.empty() &&
      mixture.strand_weights.size() != mixture.strand_counts.size()) {
    throw std::invalid_argument(std::string(description) + " strand weights have the wrong extent");
  }
  double valid_weight_sum = 0.0;
  for (std::size_t index = 0; index < mixture.strand_counts.size(); ++index) {
    const double weight = mixture.strand_weights.empty() ? 1.0 : mixture.strand_weights[index];
    if (!std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument(std::string(description) +
                                  " strand weights must be finite and nonnegative");
    }
    if (mixture.strand_counts[index] <= particle_count) {
      valid_weight_sum += weight;
    }
  }
  const bool has_valid_count =
      std::ranges::any_of(mixture.strand_counts, [particle_count](const std::size_t count) {
        return count <= particle_count;
      });
  if (has_valid_count && (!std::isfinite(valid_weight_sum) || valid_weight_sum <= 0.0)) {
    throw std::invalid_argument(std::string(description) +
                                " valid strand counts must have positive total weight");
  }
}

void validate_sweep_options(const DensityContinuationRunProvenance &provenance) {
  const std::size_t particle_count = provenance.model.free.particle_count();
  validate_fraction(provenance.sweep.segment_fraction, "segment fraction");
  validate_fraction(provenance.sweep.stitch_fraction, "sweep stitch fraction");
  validate_probability(provenance.sweep.stitch_global_partner_probability,
                       "sweep stitch global-partner probability");
  validate_stitch_mixture(provenance.sweep.stitch_mixture, particle_count, "sweep stitch mixture");
  validate_fraction(provenance.random_seam_stitch.fraction, "random-seam stitch fraction");
  validate_probability(provenance.random_seam_stitch.global_partner_probability,
                       "random-seam stitch global-partner probability");
  validate_stitch_mixture(provenance.random_seam_stitch.mixture, particle_count,
                          "random-seam stitch mixture");
}

void validate_program_metadata(const DensityContinuationRunProvenance &provenance) {
  validate_tsv_atom(provenance.program, "continuation program");
  validate_tsv_atom(provenance.program_version, "continuation program version");
}

void validate_provenance(const DensityContinuationRunProvenance &provenance) {
  provenance.model.validate();
  if (provenance.thinning_sweeps == 0) {
    throw std::invalid_argument("continuation thinning sweeps must be positive");
  }
  validate_sweep_options(provenance);
  validate_program_metadata(provenance);
}

struct ValidatedBundle {
  bool has_exact_constraints;
  std::size_t point_count;
  std::size_t value_rows;
  std::size_t covariance_rows;
  std::size_t block_rows;
  std::size_t sampler_sweeps_per_block;
};

template <class Geometry> bool validate_momentum_roles(const Geometry &geometry) {
  bool has_measured_momentum = false;
  bool has_exact_constraints = false;
  for (std::size_t momentum = 0; momentum < geometry.momentum_count(); ++momentum) {
    if (is_zero_momentum(geometry, momentum)) {
      has_exact_constraints = true;
    } else {
      has_measured_momentum = true;
    }
  }
  if (!has_measured_momentum) {
    throw std::invalid_argument("a continuation bundle requires at least one nonzero momentum");
  }
  return has_exact_constraints;
}

void validate_frequencies(const MatsubaraModeSet &modes) {
  for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
    if (modes.frequency_index(frequency) < 0) {
      throw std::invalid_argument("a continuation bundle requires nonnegative Matsubara indices");
    }
    if (!std::isfinite(modes.frequency(frequency))) {
      throw std::invalid_argument("a continuation bundle has a non-finite Matsubara frequency");
    }
  }
}

template <class Series>
void validate_value_rows(const Series &series, const std::size_t momentum, const bool exact,
                         const bool values_may_be_negative) {
  for (std::size_t point = 0; point < series_point_count(series); ++point) {
    const double mean = series_mean(series, point, momentum);
    const double error = series_standard_error(series, point, momentum);
    if (!std::isfinite(mean) || (!values_may_be_negative && mean < 0.0) || !std::isfinite(error) ||
        error < 0.0) {
      throw std::invalid_argument("a continuation value or standard error is invalid");
    }
    if (exact && (mean != 0.0 || error != 0.0)) {
      throw std::invalid_argument("fixed-particle-number q=0 rows must be exact zero");
    }
    for (std::size_t block = 0; block < series.block_count(); ++block) {
      const double value = series_block_value(series, block, point, momentum);
      if (!std::isfinite(value) || (!values_may_be_negative && value < 0.0) ||
          (exact && value != 0.0)) {
        throw std::invalid_argument("a continuation block value is invalid");
      }
    }
  }
}

template <class Series>
void validate_covariance_rows(const Series &series, const std::size_t momentum, const bool exact) {
  for (std::size_t left = 0; left < series_point_count(series); ++left) {
    for (std::size_t right = 0; right < series_point_count(series); ++right) {
      const double covariance = series_covariance(series, momentum, left, right);
      const std::array<std::size_t, 2> transposed{right, left};
      if (!std::isfinite(covariance) ||
          covariance !=
              series_covariance(series, momentum, transposed.front(), transposed.back()) ||
          (left == right && covariance < 0.0) || (exact && covariance != 0.0)) {
        throw std::invalid_argument("a continuation covariance matrix is invalid");
      }
    }
  }
}

template <class Series>
ValidatedBundle validate_bundle(const Series &series,
                                const DensityContinuationRunProvenance &provenance) {
  validate_provenance(provenance);
  if (series.model() != provenance.model.free) {
    throw std::invalid_argument(
        "continuation interacting model and density block series have different free models");
  }

  const auto &geometry = series_geometry(series);
  const BundleBasisMetadata metadata = bundle_metadata(series);
  const bool has_exact_constraints = validate_momentum_roles(geometry);
  if constexpr (std::is_same_v<Series, DensityMatsubaraBlockSeries>) {
    validate_frequencies(geometry);
  }
  for (std::size_t momentum = 0; momentum < geometry.momentum_count(); ++momentum) {
    const bool exact = is_zero_momentum(geometry, momentum);
    validate_value_rows(series, momentum, exact, metadata.values_may_be_negative);
    validate_covariance_rows(series, momentum, exact);
  }
  const std::size_t point_count = series_point_count(series);
  const std::size_t value_rows = checked_product(geometry.momentum_count(), point_count,
                                                 "continuation value row count exceeds size_t");
  const std::size_t point_square = checked_product(
      point_count, point_count, "continuation covariance point extent exceeds size_t");
  const std::size_t covariance_rows = checked_product(
      geometry.momentum_count(), point_square, "continuation covariance row count exceeds size_t");
  const std::size_t block_rows = checked_product(series.block_count(), value_rows,
                                                 "continuation block row count exceeds size_t");
  const std::size_t sampler_sweeps_per_block =
      checked_product(provenance.thinning_sweeps, series.measurements_per_block(),
                      "continuation sweeps per block exceed size_t");
  return {
      .has_exact_constraints = has_exact_constraints,
      .point_count = point_count,
      .value_rows = value_rows,
      .covariance_rows = covariance_rows,
      .block_rows = block_rows,
      .sampler_sweeps_per_block = sampler_sweeps_per_block,
  };
}

std::string optional_count(const std::optional<std::size_t> value) {
  return value.has_value() ? std::to_string(*value) : "particle_count_default";
}

template <class T> std::string comma_separated(const std::vector<T> &values) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  return output.str();
}

void configure_table_stream(std::ofstream &output) {
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
}

template <class Series>
void write_manifest(const std::filesystem::path &path, const Series &series,
                    const DensityContinuationRunProvenance &provenance,
                    const ValidatedBundle validated) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create continuation manifest: " + path.string());
  }
  configure_table_stream(output);
  const Model &model = provenance.model.free;
  const auto &geometry = series_geometry(series);
  const BundleBasisMetadata metadata = bundle_metadata(series);
  const std::size_t covariance_rank_upper_bound =
      std::min(validated.point_count, series.block_count() - 1);
  const bool full_rank_possible = series.block_count() > validated.point_count;
  const char *const row_roles =
      validated.has_exact_constraints ? "measured,exact_constraint" : "measured";

  output << "key\tvalue\n"
         << "schema_id\t" << kSchemaIdentifier << '\n'
         << "schema_version\t" << kSchemaVersion << '\n'
         << "basis\t" << metadata.basis << '\n'
         << "observable_id\tconnected_density_per_site\n"
         << "kernel_convention_id\tpositive_density_spectrum_bosonic_v1\n"
         << "fourier_spatial_phase\texp(-i*q*x)\n"
         << "fourier_temporal_phase\t" << metadata.temporal_phase << '\n'
         << "normalization_id\t" << metadata.normalization << '\n'
         << "units_convention\thbar=1;k_B=1;lattice_spacing=1\n"
         << "value_units\t" << metadata.value_units << '\n'
         << metadata.point_units_key << '\t' << metadata.point_units << '\n'
         << "model_particle_count\t" << model.particle_count() << '\n'
         << "model_beta\t" << model.beta() << '\n'
         << "model_linear_size\t" << model.linear_size() << '\n'
         << "model_dimension\t" << model.dimension() << '\n'
         << "model_hopping\t" << model.hopping() << '\n'
         << "model_interaction\t" << provenance.model.interaction << '\n'
         << "lattice_volume\t" << model.volume() << '\n'
         << "seed\t" << provenance.seed << '\n'
         << "burn_in_sweeps\t" << provenance.burn_in_sweeps << '\n'
         << "thinning_sweeps\t" << provenance.thinning_sweeps << '\n'
         << "measurement_advance_order\trandom_seam_stitch_then_sampler_sweep\n"
         << "sweep_segment_updates\t" << optional_count(provenance.sweep.segment_updates) << '\n'
         << "sweep_segment_fraction\t" << provenance.sweep.segment_fraction << '\n'
         << "sweep_cycle_updates\t" << provenance.sweep.cycle_updates << '\n'
         << "sweep_global_updates\t" << provenance.sweep.global_updates << '\n'
         << "sweep_stitch_updates\t" << provenance.sweep.stitch_updates << '\n'
         << "sweep_stitch_fraction\t" << provenance.sweep.stitch_fraction << '\n'
         << "sweep_stitch_locality_radius\t" << provenance.sweep.stitch_locality_radius << '\n'
         << "sweep_stitch_global_partner_probability\t"
         << provenance.sweep.stitch_global_partner_probability << '\n'
         << "sweep_stitch_strand_counts\t"
         << comma_separated(provenance.sweep.stitch_mixture.strand_counts) << '\n'
         << "sweep_stitch_strand_weights\t"
         << (provenance.sweep.stitch_mixture.strand_weights.empty()
                 ? "equal"
                 : comma_separated(provenance.sweep.stitch_mixture.strand_weights))
         << '\n'
         << "sweep_time_shift_updates\t" << provenance.sweep.time_shift_updates << '\n'
         << "random_seam_stitch_updates\t" << optional_count(provenance.random_seam_stitch.updates)
         << '\n'
         << "random_seam_stitch_fraction\t" << provenance.random_seam_stitch.fraction << '\n'
         << "random_seam_stitch_locality_radius\t" << provenance.random_seam_stitch.locality_radius
         << '\n'
         << "random_seam_stitch_global_partner_probability\t"
         << provenance.random_seam_stitch.global_partner_probability << '\n'
         << "random_seam_stitch_strand_counts\t"
         << comma_separated(provenance.random_seam_stitch.mixture.strand_counts) << '\n'
         << "random_seam_stitch_strand_weights\t"
         << (provenance.random_seam_stitch.mixture.strand_weights.empty()
                 ? "equal"
                 : comma_separated(provenance.random_seam_stitch.mixture.strand_weights))
         << '\n'
         << "measurements_per_block\t" << series.measurements_per_block() << '\n'
         << "completed_block_count\t" << series.block_count() << '\n'
         << "sample_count\t" << series.sample_count() << '\n'
         << "post_burn_in_sampler_sweeps_per_block\t" << validated.sampler_sweeps_per_block << '\n'
         << "momentum_count\t" << geometry.momentum_count() << '\n'
         << metadata.point_count_key << '\t' << validated.point_count << '\n'
         << "values_row_count\t" << validated.value_rows << '\n'
         << "covariance_row_count\t" << validated.covariance_rows << '\n'
         << "blocks_row_count\t" << validated.block_rows << '\n'
         << "row_roles_present\t" << row_roles << '\n'
         << "exact_constraint_rule\tfixed_particle_number_q_zero\n"
         << "covariance_kind\tstatistical_covariance_of_mean\n"
         << "covariance_scope\t" << metadata.covariance_scope << '\n'
         << "covariance_rank_upper_bound\t" << covariance_rank_upper_bound << '\n'
         << "covariance_full_rank_possible\t" << (full_rank_possible ? "true" : "false") << '\n'
         << "covariance_rank_status\t"
         << (full_rank_possible ? "full_rank_possible" : "rank_deficient_by_completed_block_count")
         << '\n'
         << "covariance_regularization\tnone\n"
         << "jackknife_representation\tleave_one_block_out_reconstructible_from_blocks\n"
         << "scalar_trace_retained\t" << (provenance.scalar_trace_retained ? "true" : "false")
         << '\n'
         << "program\t" << provenance.program << '\n'
         << "program_version\t" << provenance.program_version << '\n';
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write continuation manifest: " + path.string());
  }
}

void write_values(const std::filesystem::path &path, const DensityMatsubaraBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create continuation values: " + path.string());
  }
  configure_table_stream(output);
  const MatsubaraModeSet &modes = series.modes();
  output << "momentum";
  for (std::size_t axis = 0; axis < modes.layout().dimension(); ++axis) {
    output << "\tk_" << axis;
  }
  output << "\tfrequency\tn\tomega\tmean\tstandard_error\texact_constraint\n";
  for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
    const bool exact = is_zero_momentum(modes, momentum);
    for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
      output << momentum;
      for (const std::size_t component : modes.momentum_indices(momentum)) {
        output << '\t' << component;
      }
      output << '\t' << frequency << '\t' << modes.frequency_index(frequency) << '\t'
             << modes.frequency(frequency) << '\t' << series.mean(frequency, momentum) << '\t'
             << series.standard_error(frequency, momentum) << '\t' << (exact ? 1 : 0) << '\n';
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write continuation values: " + path.string());
  }
}

void write_values(const std::filesystem::path &path, const DensityLagBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create continuation values: " + path.string());
  }
  configure_table_stream(output);
  const ImaginaryTimeLagSet &lags = series.lags();
  output << "momentum";
  for (std::size_t axis = 0; axis < lags.layout().dimension(); ++axis) {
    output << "\tk_" << axis;
  }
  output << "\tlag\ttau\tmean\tstandard_error\texact_constraint\n";
  for (std::size_t momentum = 0; momentum < lags.momentum_count(); ++momentum) {
    const bool exact = is_zero_momentum(lags, momentum);
    for (std::size_t lag = 0; lag < lags.lag_count(); ++lag) {
      output << momentum;
      for (const std::size_t component : lags.momentum_indices(momentum)) {
        output << '\t' << component;
      }
      output << '\t' << lag << '\t' << lags.lag(lag) << '\t' << series.mean(lag, momentum) << '\t'
             << series.standard_error(lag, momentum) << '\t' << (exact ? 1 : 0) << '\n';
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write continuation values: " + path.string());
  }
}

template <class Series>
void write_covariance(const std::filesystem::path &path, const Series &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create continuation covariance: " + path.string());
  }
  configure_table_stream(output);
  const auto &geometry = series_geometry(series);
  const std::size_t point_count = series_point_count(series);
  output << "momentum\tleft\tright\tcovariance_of_mean\n";
  for (std::size_t momentum = 0; momentum < geometry.momentum_count(); ++momentum) {
    for (std::size_t left = 0; left < point_count; ++left) {
      for (std::size_t right = 0; right < point_count; ++right) {
        output << momentum << '\t' << left << '\t' << right << '\t'
               << series_covariance(series, momentum, left, right) << '\n';
      }
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write continuation covariance: " + path.string());
  }
}

template <class Series> void write_blocks(const std::filesystem::path &path, const Series &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create continuation blocks: " + path.string());
  }
  configure_table_stream(output);
  const auto &geometry = series_geometry(series);
  const std::size_t point_count = series_point_count(series);
  output << "block\tmomentum\tfrequency_or_lag\tvalue\n";
  for (std::size_t block = 0; block < series.block_count(); ++block) {
    for (std::size_t momentum = 0; momentum < geometry.momentum_count(); ++momentum) {
      for (std::size_t point = 0; point < point_count; ++point) {
        output << block << '\t' << momentum << '\t' << point << '\t'
               << series_block_value(series, block, point, momentum) << '\n';
      }
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write continuation blocks: " + path.string());
  }
}

struct ValidatedHoppingBundle {
  std::size_t response_rows;
  std::size_t response_block_rows;
  std::size_t mean_flux_rows;
  std::size_t mean_flux_block_rows;
  std::size_t diamagnetic_rows;
  std::size_t diamagnetic_block_rows;
  std::size_t sampler_sweeps_per_block;
};

ValidatedHoppingBundle validate_hopping_bundle(const HoppingResponseBlockSeries &series,
                                               const HoppingResponseRunProvenance &provenance) {
  validate_provenance(provenance);
  if (series.model() != provenance.model.free) {
    throw std::invalid_argument("hopping response series and run provenance models differ");
  }
  const std::size_t dimension = series.model().dimension();
  const std::size_t mean_flux_rows = checked_product(series.modes().mode_count(), dimension,
                                                     "hopping mean-flux row count exceeds size_t");
  const std::size_t response_rows =
      checked_product(mean_flux_rows, dimension, "hopping response row count exceeds size_t");
  const std::size_t diamagnetic_rows = dimension;
  const std::size_t sampler_sweeps_per_block =
      checked_product(provenance.thinning_sweeps, series.measurements_per_block(),
                      "hopping response sweeps per block exceed size_t");
  return {
      .response_rows = response_rows,
      .response_block_rows = checked_product(series.block_count(), response_rows,
                                             "hopping response block row count exceeds size_t"),
      .mean_flux_rows = mean_flux_rows,
      .mean_flux_block_rows = checked_product(series.block_count(), mean_flux_rows,
                                              "hopping mean-flux block row count exceeds size_t"),
      .diamagnetic_rows = diamagnetic_rows,
      .diamagnetic_block_rows =
          checked_product(series.block_count(), diamagnetic_rows,
                          "hopping diamagnetic block row count exceeds size_t"),
      .sampler_sweeps_per_block = sampler_sweeps_per_block,
  };
}

void write_hopping_manifest(const std::filesystem::path &path,
                            const HoppingResponseBlockSeries &series,
                            const HoppingResponseRunProvenance &provenance,
                            const ValidatedHoppingBundle validated) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create hopping-response manifest: " + path.string());
  }
  configure_table_stream(output);
  const Model &model = provenance.model.free;
  output << "key\tvalue\n"
         << "schema_id\thopping-response\n"
         << "schema_version\t1\n"
         << "basis\tbosonic_matsubara\n"
         << "observable_id\tfull_gauge_flux_response\n"
         << "source_convention\tpositive_bond_peierls_phase\n"
         << "spatial_phase\texp(-i*q*bond_midpoint)\n"
         << "temporal_phase\texp(+i*omega_n*tau)\n"
         << "flux_response_normalization\tmean_I_left_conj_I_right_over_beta_volume\n"
         << "diamagnetic_normalization\tmean_axis_event_count_over_beta_volume\n"
         << "paramagnetic_definition\tdiamagnetic_delta_minus_flux_response\n"
         << "units_convention\thbar=1;k_B=1;lattice_spacing=1\n"
         << "response_units\tenergy_per_site\n"
         << "mean_flux_units\tdimensionless\n"
         << "model_particle_count\t" << model.particle_count() << '\n'
         << "model_beta\t" << model.beta() << '\n'
         << "model_linear_size\t" << model.linear_size() << '\n'
         << "model_dimension\t" << model.dimension() << '\n'
         << "model_hopping\t" << model.hopping() << '\n'
         << "model_interaction\t" << provenance.model.interaction << '\n'
         << "lattice_volume\t" << model.volume() << '\n'
         << "seed\t" << provenance.seed << '\n'
         << "burn_in_sweeps\t" << provenance.burn_in_sweeps << '\n'
         << "thinning_sweeps\t" << provenance.thinning_sweeps << '\n'
         << "measurement_advance_order\trandom_seam_stitch_then_sampler_sweep\n"
         << "sweep_segment_updates\t" << optional_count(provenance.sweep.segment_updates) << '\n'
         << "sweep_segment_fraction\t" << provenance.sweep.segment_fraction << '\n'
         << "sweep_cycle_updates\t" << provenance.sweep.cycle_updates << '\n'
         << "sweep_global_updates\t" << provenance.sweep.global_updates << '\n'
         << "sweep_stitch_updates\t" << provenance.sweep.stitch_updates << '\n'
         << "sweep_stitch_fraction\t" << provenance.sweep.stitch_fraction << '\n'
         << "sweep_stitch_locality_radius\t" << provenance.sweep.stitch_locality_radius << '\n'
         << "sweep_stitch_global_partner_probability\t"
         << provenance.sweep.stitch_global_partner_probability << '\n'
         << "sweep_stitch_strand_counts\t"
         << comma_separated(provenance.sweep.stitch_mixture.strand_counts) << '\n'
         << "sweep_stitch_strand_weights\t"
         << (provenance.sweep.stitch_mixture.strand_weights.empty()
                 ? "equal"
                 : comma_separated(provenance.sweep.stitch_mixture.strand_weights))
         << '\n'
         << "sweep_time_shift_updates\t" << provenance.sweep.time_shift_updates << '\n'
         << "random_seam_stitch_updates\t" << optional_count(provenance.random_seam_stitch.updates)
         << '\n'
         << "random_seam_stitch_fraction\t" << provenance.random_seam_stitch.fraction << '\n'
         << "random_seam_stitch_locality_radius\t" << provenance.random_seam_stitch.locality_radius
         << '\n'
         << "random_seam_stitch_global_partner_probability\t"
         << provenance.random_seam_stitch.global_partner_probability << '\n'
         << "random_seam_stitch_strand_counts\t"
         << comma_separated(provenance.random_seam_stitch.mixture.strand_counts) << '\n'
         << "random_seam_stitch_strand_weights\t"
         << (provenance.random_seam_stitch.mixture.strand_weights.empty()
                 ? "equal"
                 : comma_separated(provenance.random_seam_stitch.mixture.strand_weights))
         << '\n'
         << "measurements_per_block\t" << series.measurements_per_block() << '\n'
         << "completed_block_count\t" << series.block_count() << '\n'
         << "sample_count\t" << series.sample_count() << '\n'
         << "post_burn_in_sampler_sweeps_per_block\t" << validated.sampler_sweeps_per_block << '\n'
         << "momentum_count\t" << series.modes().momentum_count() << '\n'
         << "frequency_count\t" << series.modes().frequency_count() << '\n'
         << "response_values_row_count\t" << validated.response_rows << '\n'
         << "response_blocks_row_count\t" << validated.response_block_rows << '\n'
         << "mean_flux_values_row_count\t" << validated.mean_flux_rows << '\n'
         << "mean_flux_blocks_row_count\t" << validated.mean_flux_block_rows << '\n'
         << "diamagnetic_values_row_count\t" << validated.diamagnetic_rows << '\n'
         << "diamagnetic_blocks_row_count\t" << validated.diamagnetic_block_rows << '\n'
         << "standard_error_kind\tcompleted_block_standard_error_of_mean\n"
         << "jackknife_representation\tleave_one_block_out_reconstructible_from_blocks\n"
         << "conductivity_interpretation\tnone\n"
         << "scalar_trace_retained\t" << (provenance.scalar_trace_retained ? "true" : "false")
         << '\n'
         << "program\t" << provenance.program << '\n'
         << "program_version\t" << provenance.program_version << '\n';
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write hopping-response manifest: " + path.string());
  }
}

void write_hopping_response_values(const std::filesystem::path &path,
                                   const HoppingResponseBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create hopping-response values: " + path.string());
  }
  configure_table_stream(output);
  const MatsubaraModeSet &modes = series.modes();
  output << "momentum";
  for (std::size_t axis = 0; axis < modes.layout().dimension(); ++axis) {
    output << "\tk_" << axis;
  }
  output << "\tfrequency\tn\tomega\tleft_axis\tright_axis"
            "\tflux_response_mean_real\tflux_response_mean_imag"
            "\tflux_response_standard_error_real\tflux_response_standard_error_imag"
            "\tparamagnetic_mean_real\tparamagnetic_mean_imag"
            "\tparamagnetic_standard_error_real\tparamagnetic_standard_error_imag\n";
  for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
    for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
      for (std::size_t left = 0; left < series.model().dimension(); ++left) {
        for (std::size_t right = 0; right < series.model().dimension(); ++right) {
          const std::complex<double> response =
              series.flux_response(frequency, momentum, left, right);
          const std::complex<double> paramagnetic =
              series.paramagnetic(frequency, momentum, left, right);
          output << momentum;
          for (const std::size_t component : modes.momentum_indices(momentum)) {
            output << '\t' << component;
          }
          output << '\t' << frequency << '\t' << modes.frequency_index(frequency) << '\t'
                 << modes.frequency(frequency) << '\t' << left << '\t' << right << '\t'
                 << response.real() << '\t' << response.imag() << '\t'
                 << series.flux_response_standard_error(frequency, momentum, left, right,
                                                        HoppingResponseComponent::Real)
                 << '\t'
                 << series.flux_response_standard_error(frequency, momentum, left, right,
                                                        HoppingResponseComponent::Imaginary)
                 << '\t' << paramagnetic.real() << '\t' << paramagnetic.imag() << '\t'
                 << series.paramagnetic_standard_error(frequency, momentum, left, right,
                                                       HoppingResponseComponent::Real)
                 << '\t'
                 << series.paramagnetic_standard_error(frequency, momentum, left, right,
                                                       HoppingResponseComponent::Imaginary)
                 << '\n';
        }
      }
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write hopping-response values: " + path.string());
  }
}

void write_hopping_response_blocks(const std::filesystem::path &path,
                                   const HoppingResponseBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create hopping-response blocks: " + path.string());
  }
  configure_table_stream(output);
  output << "block\tmomentum\tfrequency\tleft_axis\tright_axis"
            "\tflux_response_real\tflux_response_imag"
            "\tparamagnetic_real\tparamagnetic_imag\n";
  for (std::size_t block = 0; block < series.block_count(); ++block) {
    for (std::size_t momentum = 0; momentum < series.modes().momentum_count(); ++momentum) {
      for (std::size_t frequency = 0; frequency < series.modes().frequency_count(); ++frequency) {
        for (std::size_t left = 0; left < series.model().dimension(); ++left) {
          for (std::size_t right = 0; right < series.model().dimension(); ++right) {
            const std::complex<double> response =
                series.block_flux_response(block, frequency, momentum, left, right);
            const std::complex<double> paramagnetic =
                series.block_paramagnetic(block, frequency, momentum, left, right);
            output << block << '\t' << momentum << '\t' << frequency << '\t' << left << '\t'
                   << right << '\t' << response.real() << '\t' << response.imag() << '\t'
                   << paramagnetic.real() << '\t' << paramagnetic.imag() << '\n';
          }
        }
      }
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write hopping-response blocks: " + path.string());
  }
}

void write_hopping_mean_flux_values(const std::filesystem::path &path,
                                    const HoppingResponseBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create hopping mean-flux values: " + path.string());
  }
  configure_table_stream(output);
  const MatsubaraModeSet &modes = series.modes();
  output << "momentum";
  for (std::size_t axis = 0; axis < modes.layout().dimension(); ++axis) {
    output << "\tk_" << axis;
  }
  output << "\tfrequency\tn\tomega\taxis\tmean_real\tmean_imag"
            "\tstandard_error_real\tstandard_error_imag\n";
  for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
    for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
      for (std::size_t axis = 0; axis < series.model().dimension(); ++axis) {
        const std::complex<double> mean = series.mean_flux(frequency, momentum, axis);
        output << momentum;
        for (const std::size_t component : modes.momentum_indices(momentum)) {
          output << '\t' << component;
        }
        output << '\t' << frequency << '\t' << modes.frequency_index(frequency) << '\t'
               << modes.frequency(frequency) << '\t' << axis << '\t' << mean.real() << '\t'
               << mean.imag() << '\t'
               << series.mean_flux_standard_error(frequency, momentum, axis,
                                                  HoppingResponseComponent::Real)
               << '\t'
               << series.mean_flux_standard_error(frequency, momentum, axis,
                                                  HoppingResponseComponent::Imaginary)
               << '\n';
      }
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write hopping mean-flux values: " + path.string());
  }
}

void write_hopping_mean_flux_blocks(const std::filesystem::path &path,
                                    const HoppingResponseBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create hopping mean-flux blocks: " + path.string());
  }
  configure_table_stream(output);
  output << "block\tmomentum\tfrequency\taxis\tmean_real\tmean_imag\n";
  for (std::size_t block = 0; block < series.block_count(); ++block) {
    for (std::size_t momentum = 0; momentum < series.modes().momentum_count(); ++momentum) {
      for (std::size_t frequency = 0; frequency < series.modes().frequency_count(); ++frequency) {
        for (std::size_t axis = 0; axis < series.model().dimension(); ++axis) {
          const std::complex<double> mean =
              series.block_mean_flux(block, frequency, momentum, axis);
          output << block << '\t' << momentum << '\t' << frequency << '\t' << axis << '\t'
                 << mean.real() << '\t' << mean.imag() << '\n';
        }
      }
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write hopping mean-flux blocks: " + path.string());
  }
}

void write_hopping_diamagnetic_values(const std::filesystem::path &path,
                                      const HoppingResponseBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create hopping diamagnetic values: " + path.string());
  }
  configure_table_stream(output);
  output << "axis\tmean\tstandard_error\n";
  for (std::size_t axis = 0; axis < series.model().dimension(); ++axis) {
    output << axis << '\t' << series.diamagnetic(axis) << '\t'
           << series.diamagnetic_standard_error(axis) << '\n';
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write hopping diamagnetic values: " + path.string());
  }
}

void write_hopping_diamagnetic_blocks(const std::filesystem::path &path,
                                      const HoppingResponseBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create hopping diamagnetic blocks: " + path.string());
  }
  configure_table_stream(output);
  output << "block\taxis\tvalue\n";
  for (std::size_t block = 0; block < series.block_count(); ++block) {
    for (std::size_t axis = 0; axis < series.model().dimension(); ++axis) {
      output << block << '\t' << axis << '\t' << series.block_diamagnetic(block, axis) << '\n';
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write hopping diamagnetic blocks: " + path.string());
  }
}

std::filesystem::path create_temporary_directory(const std::filesystem::path &destination) {
  constexpr std::size_t kMaximumAttempts = 1024;
  for (std::size_t attempt = 0; attempt < kMaximumAttempts; ++attempt) {
    std::filesystem::path candidate = destination;
    candidate += ".tmp." + std::to_string(attempt);
    std::error_code error;
    if (std::filesystem::create_directory(candidate, error)) {
      return candidate;
    }
    if (error && error != std::errc::file_exists) {
      throw std::filesystem::filesystem_error("failed to create continuation temporary directory",
                                              candidate, error);
    }
  }
  throw std::runtime_error("could not reserve a continuation temporary directory beside: " +
                           destination.string());
}

[[noreturn]] void report_cleanup_failure(const std::filesystem::path &temporary,
                                         const std::string &original,
                                         const std::error_code cleanup_error) {
  throw std::runtime_error(original + "; additionally failed to remove temporary bundle " +
                           temporary.string() + ": " + cleanup_error.message());
}

} // namespace

void validate_density_continuation_bundle_destination(const std::filesystem::path &destination) {
  if (destination.empty()) {
    throw std::invalid_argument("continuation bundle destination must not be empty");
  }
  const std::filesystem::path filename = destination.filename();
  if (filename.empty() || filename == "." || filename == "..") {
    throw std::invalid_argument("continuation bundle destination must name a directory");
  }

  std::error_code error;
  std::filesystem::file_status destination_status =
      std::filesystem::symlink_status(destination, error);
  if (error == std::errc::no_such_file_or_directory) {
    error.clear();
    destination_status = std::filesystem::file_status(std::filesystem::file_type::not_found);
  } else if (error) {
    throw std::filesystem::filesystem_error("cannot inspect continuation bundle destination",
                                            destination, error);
  }
  if (destination_status.type() != std::filesystem::file_type::not_found) {
    throw std::invalid_argument("continuation bundle destination already exists: " +
                                destination.string());
  }

  std::filesystem::path parent = destination.parent_path();
  if (parent.empty()) {
    parent = ".";
  }
  const std::filesystem::file_status parent_status = std::filesystem::status(parent, error);
  if (error) {
    throw std::filesystem::filesystem_error("cannot inspect continuation bundle parent", parent,
                                            error);
  }
  if (!std::filesystem::is_directory(parent_status)) {
    throw std::invalid_argument("continuation bundle parent is not a directory: " +
                                parent.string());
  }
}

template <class Series>
void write_bundle(const std::filesystem::path &destination, const Series &series,
                  const DensityContinuationRunProvenance &provenance) {
  const ValidatedBundle validated = validate_bundle(series, provenance);
  validate_density_continuation_bundle_destination(destination);

  const std::filesystem::path temporary = create_temporary_directory(destination);
  try {
    write_manifest(temporary / "manifest.tsv", series, provenance, validated);
    write_values(temporary / "values.tsv", series);
    write_covariance(temporary / "covariance.tsv", series);
    write_blocks(temporary / "blocks.tsv", series);
    validate_density_continuation_bundle_destination(destination);
    std::filesystem::rename(temporary, destination);
  } catch (const std::exception &error) {
    const std::string original = error.what();
    std::error_code cleanup_error;
    std::filesystem::remove_all(temporary, cleanup_error);
    if (cleanup_error) {
      report_cleanup_failure(temporary, original, cleanup_error);
    }
    throw;
  } catch (...) {
    std::error_code cleanup_error;
    std::filesystem::remove_all(temporary, cleanup_error);
    if (cleanup_error) {
      report_cleanup_failure(temporary, "unknown continuation bundle publication failure",
                             cleanup_error);
    }
    throw;
  }
}

void write_density_continuation_bundle(const std::filesystem::path &destination,
                                       const DensityMatsubaraBlockSeries &series,
                                       const DensityContinuationRunProvenance &provenance) {
  write_bundle(destination, series, provenance);
}

void write_density_continuation_bundle(const std::filesystem::path &destination,
                                       const DensityLagBlockSeries &series,
                                       const DensityContinuationRunProvenance &provenance) {
  write_bundle(destination, series, provenance);
}

void validate_hopping_response_bundle_destination(const std::filesystem::path &destination) {
  validate_density_continuation_bundle_destination(destination);
}

void write_hopping_response_bundle(const std::filesystem::path &destination,
                                   const HoppingResponseBlockSeries &series,
                                   const HoppingResponseRunProvenance &provenance) {
  const ValidatedHoppingBundle validated = validate_hopping_bundle(series, provenance);
  validate_hopping_response_bundle_destination(destination);

  const std::filesystem::path temporary = create_temporary_directory(destination);
  try {
    write_hopping_manifest(temporary / "manifest.tsv", series, provenance, validated);
    write_hopping_response_values(temporary / "response_values.tsv", series);
    write_hopping_response_blocks(temporary / "response_blocks.tsv", series);
    write_hopping_mean_flux_values(temporary / "mean_flux_values.tsv", series);
    write_hopping_mean_flux_blocks(temporary / "mean_flux_blocks.tsv", series);
    write_hopping_diamagnetic_values(temporary / "diamagnetic_values.tsv", series);
    write_hopping_diamagnetic_blocks(temporary / "diamagnetic_blocks.tsv", series);
    validate_hopping_response_bundle_destination(destination);
    std::filesystem::rename(temporary, destination);
  } catch (const std::exception &error) {
    const std::string original = error.what();
    std::error_code cleanup_error;
    std::filesystem::remove_all(temporary, cleanup_error);
    if (cleanup_error) {
      report_cleanup_failure(temporary, original, cleanup_error);
    }
    throw;
  } catch (...) {
    std::error_code cleanup_error;
    std::filesystem::remove_all(temporary, cleanup_error);
    if (cleanup_error) {
      report_cleanup_failure(temporary, "unknown hopping-response publication failure",
                             cleanup_error);
    }
    throw;
  }
}

} // namespace qmc::example
