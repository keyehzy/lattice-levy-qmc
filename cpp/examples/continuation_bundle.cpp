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
#include <utility>
#include <vector>

namespace qmc::example {
namespace {

constexpr std::string_view kSchemaIdentifier = "density-continuation";
constexpr std::string_view kSchemaVersion = "1";

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

bool is_zero_momentum(const MatsubaraModeSet &modes, const std::size_t momentum) {
  return std::ranges::all_of(modes.momentum_indices(momentum),
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
  std::size_t value_rows;
  std::size_t covariance_rows;
  std::size_t block_rows;
  std::size_t sampler_sweeps_per_block;
};

bool validate_momentum_roles(const MatsubaraModeSet &modes) {
  bool has_measured_momentum = false;
  bool has_exact_constraints = false;
  for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
    if (is_zero_momentum(modes, momentum)) {
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

void validate_value_rows(const DensityMatsubaraBlockSeries &series, const std::size_t momentum,
                         const bool exact) {
  for (std::size_t frequency = 0; frequency < series.modes().frequency_count(); ++frequency) {
    const double mean = series.mean(frequency, momentum);
    const double error = series.standard_error(frequency, momentum);
    if (!std::isfinite(mean) || mean < 0.0 || !std::isfinite(error) || error < 0.0) {
      throw std::invalid_argument("a continuation value or standard error is invalid");
    }
    if (exact && (mean != 0.0 || error != 0.0)) {
      throw std::invalid_argument("fixed-particle-number q=0 rows must be exact zero");
    }
    for (std::size_t block = 0; block < series.block_count(); ++block) {
      const double value = series.block_value(block, frequency, momentum);
      if (!std::isfinite(value) || value < 0.0 || (exact && value != 0.0)) {
        throw std::invalid_argument("a continuation block value is invalid");
      }
    }
  }
}

void validate_covariance_rows(const DensityMatsubaraBlockSeries &series, const std::size_t momentum,
                              const bool exact) {
  for (std::size_t left = 0; left < series.modes().frequency_count(); ++left) {
    for (std::size_t right = 0; right < series.modes().frequency_count(); ++right) {
      const double covariance = series.covariance_of_mean(momentum, left, right);
      const std::array<std::size_t, 2> transposed{right, left};
      if (!std::isfinite(covariance) ||
          covariance !=
              series.covariance_of_mean(momentum, transposed.front(), transposed.back()) ||
          (left == right && covariance < 0.0) || (exact && covariance != 0.0)) {
        throw std::invalid_argument("a continuation covariance matrix is invalid");
      }
    }
  }
}

ValidatedBundle validate_bundle(const DensityMatsubaraBlockSeries &series,
                                const DensityContinuationRunProvenance &provenance) {
  validate_provenance(provenance);
  if (series.model() != provenance.model.free) {
    throw std::invalid_argument(
        "continuation interacting model and density block series have different free models");
  }

  const MatsubaraModeSet &modes = series.modes();
  const bool has_exact_constraints = validate_momentum_roles(modes);
  validate_frequencies(modes);
  for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
    const bool exact = is_zero_momentum(modes, momentum);
    validate_value_rows(series, momentum, exact);
    validate_covariance_rows(series, momentum, exact);
  }
  const std::size_t value_rows = checked_product(modes.momentum_count(), modes.frequency_count(),
                                                 "continuation value row count exceeds size_t");
  const std::size_t frequency_square =
      checked_product(modes.frequency_count(), modes.frequency_count(),
                      "continuation covariance frequency extent exceeds size_t");
  const std::size_t covariance_rows = checked_product(
      modes.momentum_count(), frequency_square, "continuation covariance row count exceeds size_t");
  const std::size_t block_rows = checked_product(series.block_count(), value_rows,
                                                 "continuation block row count exceeds size_t");
  const std::size_t sampler_sweeps_per_block =
      checked_product(provenance.thinning_sweeps, series.measurements_per_block(),
                      "continuation sweeps per block exceed size_t");
  return {
      .has_exact_constraints = has_exact_constraints,
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

void write_manifest(const std::filesystem::path &path, const DensityMatsubaraBlockSeries &series,
                    const DensityContinuationRunProvenance &provenance,
                    const ValidatedBundle validated) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create continuation manifest: " + path.string());
  }
  configure_table_stream(output);
  const Model &model = provenance.model.free;
  const std::size_t covariance_rank_upper_bound =
      std::min(series.modes().frequency_count(), series.block_count() - 1);
  const bool full_rank_possible = series.block_count() > series.modes().frequency_count();
  const char *const row_roles =
      validated.has_exact_constraints ? "measured,exact_constraint" : "measured";

  output << "key\tvalue\n"
         << "schema_id\t" << kSchemaIdentifier << '\n'
         << "schema_version\t" << kSchemaVersion << '\n'
         << "basis\tbosonic_matsubara\n"
         << "observable_id\tconnected_density_per_site\n"
         << "kernel_convention_id\tpositive_density_spectrum_bosonic_v1\n"
         << "fourier_spatial_phase\texp(-i*q*x)\n"
         << "fourier_temporal_phase\texp(+i*omega_n*tau)\n"
         << "normalization_id\tmean_abs_centered_density_amplitude_squared_over_beta_volume\n"
         << "units_convention\thbar=1;k_B=1;lattice_spacing=1\n"
         << "value_units\tinverse_energy_per_site\n"
         << "frequency_units\tenergy\n"
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
         << "values_row_count\t" << validated.value_rows << '\n'
         << "covariance_row_count\t" << validated.covariance_rows << '\n'
         << "blocks_row_count\t" << validated.block_rows << '\n'
         << "row_roles_present\t" << row_roles << '\n'
         << "exact_constraint_rule\tfixed_particle_number_q_zero\n"
         << "covariance_kind\tstatistical_covariance_of_mean\n"
         << "covariance_scope\tindependent_dense_frequency_matrix_per_momentum\n"
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

void write_covariance(const std::filesystem::path &path,
                      const DensityMatsubaraBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create continuation covariance: " + path.string());
  }
  configure_table_stream(output);
  const MatsubaraModeSet &modes = series.modes();
  output << "momentum\tleft\tright\tcovariance_of_mean\n";
  for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
    for (std::size_t left = 0; left < modes.frequency_count(); ++left) {
      for (std::size_t right = 0; right < modes.frequency_count(); ++right) {
        output << momentum << '\t' << left << '\t' << right << '\t'
               << series.covariance_of_mean(momentum, left, right) << '\n';
      }
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write continuation covariance: " + path.string());
  }
}

void write_blocks(const std::filesystem::path &path, const DensityMatsubaraBlockSeries &series) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create continuation blocks: " + path.string());
  }
  configure_table_stream(output);
  const MatsubaraModeSet &modes = series.modes();
  output << "block\tmomentum\tfrequency_or_lag\tvalue\n";
  for (std::size_t block = 0; block < series.block_count(); ++block) {
    for (std::size_t momentum = 0; momentum < modes.momentum_count(); ++momentum) {
      for (std::size_t frequency = 0; frequency < modes.frequency_count(); ++frequency) {
        output << block << '\t' << momentum << '\t' << frequency << '\t'
               << series.block_value(block, frequency, momentum) << '\n';
      }
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to write continuation blocks: " + path.string());
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

void write_density_continuation_bundle(const std::filesystem::path &destination,
                                       const DensityMatsubaraBlockSeries &series,
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

} // namespace qmc::example
