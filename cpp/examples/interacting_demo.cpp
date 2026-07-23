#include "continuation_bundle.hpp"
#include "qmc/interacting.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

using DensityContinuationGeometry = std::variant<qmc::MatsubaraModeSet, qmc::ImaginaryTimeLagSet>;

struct DensityContinuationCommand {
  DensityContinuationGeometry geometry;
  std::size_t measurements_per_block;
  std::filesystem::path output_directory;
};

struct HoppingResponseCommand {
  qmc::MatsubaraModeSet modes;
  std::size_t measurements_per_block;
  std::filesystem::path output_directory;
};

struct CommandLine {
  qmc::InteractingModel model;
  std::uint64_t seed;
  std::size_t samples;
  std::size_t burn_in;
  std::size_t thin;
  qmc::SweepOptions sweep;
  qmc::RandomSeamStitchOptions random_seam_stitch;
  std::filesystem::path output;
  bool retain_scalar_trace;
  std::optional<DensityContinuationCommand> density_continuation;
  std::optional<HoppingResponseCommand> hopping_response;
};

std::string_view trim(const std::string_view value) {
  std::size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::size_t parse_momentum_component(const std::string_view input) {
  const std::string_view value = trim(input);
  if (value.empty()) {
    throw std::invalid_argument("momentum contains an empty component");
  }
  std::size_t component = 0;
  const auto conversion = std::from_chars(value.data(), value.data() + value.size(), component);
  if (conversion.ec != std::errc{} || conversion.ptr != value.data() + value.size()) {
    throw std::invalid_argument("momentum components must be nonnegative integers");
  }
  return component;
}

std::vector<std::vector<std::size_t>> parse_momentum_rows(const std::string_view input,
                                                          const std::size_t dimension) {
  if (trim(input).empty()) {
    throw std::invalid_argument("momenta must not be empty");
  }
  std::vector<std::vector<std::size_t>> momenta;
  std::size_t row_begin = 0;
  while (row_begin <= input.size()) {
    const std::size_t row_end = input.find(';', row_begin);
    const std::string_view row =
        input.substr(row_begin, row_end == std::string_view::npos ? input.size() - row_begin
                                                                  : row_end - row_begin);
    if (trim(row).empty()) {
      throw std::invalid_argument("momenta contain an empty row");
    }
    std::vector<std::size_t> components;
    std::size_t component_begin = 0;
    while (component_begin <= row.size()) {
      const std::size_t component_end = row.find(',', component_begin);
      components.push_back(parse_momentum_component(
          row.substr(component_begin, component_end == std::string_view::npos
                                          ? row.size() - component_begin
                                          : component_end - component_begin)));
      if (component_end == std::string_view::npos) {
        break;
      }
      component_begin = component_end + 1;
    }
    if (components.size() != dimension) {
      throw std::invalid_argument("each momentum row must match the model dimension");
    }
    momenta.push_back(std::move(components));
    if (row_end == std::string_view::npos) {
      break;
    }
    row_begin = row_end + 1;
  }
  return momenta;
}

bool is_zero_momentum(const std::vector<std::size_t> &momentum) {
  return std::ranges::all_of(momentum, [](const std::size_t component) { return component == 0; });
}

qmc::MatsubaraModeSet make_density_matsubara_modes(const qmc::InteractingModel &model,
                                                   const qmc::TorusLayout &layout,
                                                   std::vector<std::vector<std::size_t>> momenta,
                                                   const std::size_t frequency_max) {
  if (frequency_max >
      static_cast<std::size_t>(qmc::ContinuousMatsubaraPlan::kMaximumAbsoluteFrequencyIndex)) {
    throw std::invalid_argument("density Matsubara index exceeds the continuous phase bound");
  }
  std::vector<std::int64_t> frequencies;
  if (frequency_max == std::numeric_limits<std::size_t>::max() ||
      frequency_max + 1 > frequencies.max_size()) {
    throw std::length_error("density Matsubara frequency range is too large");
  }
  frequencies.reserve(frequency_max + 1);
  for (std::size_t frequency = 0; frequency <= frequency_max; ++frequency) {
    frequencies.push_back(static_cast<std::int64_t>(frequency));
  }
  return {model.free.beta(),
          layout,
          {.momentum_indices = std::move(momenta), .frequency_indices = std::move(frequencies)}};
}

DensityContinuationGeometry make_density_geometry(const cxxopts::ParseResult &result,
                                                  const qmc::InteractingModel &model,
                                                  std::vector<std::vector<std::size_t>> momenta,
                                                  const bool has_frequency) {
  const qmc::TorusLayout layout(model.free.linear_size(), model.free.dimension());
  if (has_frequency) {
    return make_density_matsubara_modes(model, layout, std::move(momenta),
                                        result["density-frequency-max"].as<std::size_t>());
  }
  return qmc::ImaginaryTimeLagSet(model.free.beta(), layout,
                                  {.momentum_indices = std::move(momenta),
                                   .lags = result["density-lags"].as<std::vector<double>>()});
}

void validate_density_plan(const DensityContinuationGeometry &geometry) {
  std::visit(
      [](const auto &requested_geometry) {
        using Geometry = std::remove_cvref_t<decltype(requested_geometry)>;
        if constexpr (std::is_same_v<Geometry, qmc::MatsubaraModeSet>) {
          static_cast<void>(qmc::ContinuousMatsubaraPlan(requested_geometry));
        } else {
          static_cast<void>(qmc::ContinuousDensityLagPlan(requested_geometry));
        }
      },
      geometry);
}

std::optional<DensityContinuationCommand>
parse_density_continuation(const cxxopts::ParseResult &result, const qmc::InteractingModel &model,
                           const std::size_t samples) {
  const bool has_destination = result.contains("density-continuation-dir");
  const bool has_momenta = result.contains("density-momenta");
  const bool has_frequency = result.contains("density-frequency-max");
  const bool has_lags = result.contains("density-lags");
  const bool has_block_size = result.contains("density-measurements-per-block");
  if (!has_destination) {
    if (has_momenta || has_frequency || has_lags || has_block_size) {
      throw std::invalid_argument(
          "density continuation options require --density-continuation-dir");
    }
    return std::nullopt;
  }
  if (!has_momenta || !has_block_size || (!has_frequency && !has_lags)) {
    throw std::invalid_argument("density continuation requires --density-momenta, "
                                "--density-measurements-per-block, and exactly one of "
                                "--density-frequency-max or --density-lags");
  }
  if (has_frequency && has_lags) {
    throw std::invalid_argument(
        "--density-frequency-max and --density-lags select different continuation bases");
  }

  std::vector<std::vector<std::size_t>> momenta =
      parse_momentum_rows(result["density-momenta"].as<std::string>(), model.free.dimension());
  if (std::ranges::all_of(momenta, is_zero_momentum)) {
    throw std::invalid_argument("density continuation requires at least one nonzero momentum row");
  }
  const std::size_t measurements_per_block =
      result["density-measurements-per-block"].as<std::size_t>();
  if (measurements_per_block == 0) {
    throw std::invalid_argument("density measurements per block must be positive");
  }
  if (samples % measurements_per_block != 0) {
    throw std::invalid_argument(
        "density continuation samples must form complete equal-size blocks");
  }
  if (samples / measurements_per_block < 2) {
    throw std::invalid_argument("density continuation requires at least two complete blocks");
  }

  DensityContinuationCommand command{
      .geometry = make_density_geometry(result, model, std::move(momenta), has_frequency),
      .measurements_per_block = measurements_per_block,
      .output_directory = result["density-continuation-dir"].as<std::string>(),
  };
  validate_density_plan(command.geometry);
  return command;
}

std::optional<HoppingResponseCommand> parse_hopping_response(const cxxopts::ParseResult &result,
                                                             const qmc::InteractingModel &model,
                                                             const std::size_t samples) {
  const bool has_destination = result.contains("hopping-response-dir");
  const bool has_momenta = result.contains("hopping-momenta");
  const bool has_frequency = result.contains("hopping-frequency-max");
  const bool has_block_size = result.contains("hopping-measurements-per-block");
  if (!has_destination) {
    if (has_momenta || has_frequency || has_block_size) {
      throw std::invalid_argument("hopping response options require --hopping-response-dir");
    }
    return std::nullopt;
  }
  if (!has_momenta || !has_frequency || !has_block_size) {
    throw std::invalid_argument("hopping response requires --hopping-momenta, "
                                "--hopping-frequency-max, and "
                                "--hopping-measurements-per-block");
  }
  std::vector<std::vector<std::size_t>> momenta =
      parse_momentum_rows(result["hopping-momenta"].as<std::string>(), model.free.dimension());
  const std::size_t measurements_per_block =
      result["hopping-measurements-per-block"].as<std::size_t>();
  if (measurements_per_block == 0) {
    throw std::invalid_argument("hopping measurements per block must be positive");
  }
  if (samples % measurements_per_block != 0) {
    throw std::invalid_argument("hopping response samples must form complete equal-size blocks");
  }
  if (samples / measurements_per_block < 2) {
    throw std::invalid_argument("hopping response requires at least two complete blocks");
  }
  const qmc::TorusLayout layout(model.free.linear_size(), model.free.dimension());
  HoppingResponseCommand command{
      .modes = make_density_matsubara_modes(model, layout, std::move(momenta),
                                            result["hopping-frequency-max"].as<std::size_t>()),
      .measurements_per_block = measurements_per_block,
      .output_directory = result["hopping-response-dir"].as<std::string>(),
  };
  static_cast<void>(qmc::ContinuousMatsubaraPlan(command.modes));
  return command;
}

std::optional<CommandLine> parse_command_line(const int argc, char **argv) {
  cxxopts::Options options(argv[0],
                           "Run the canonical continuous-time finite-U Bose-Hubbard sampler");
  options.add_options(
      "",
      {{"n,particles", "Particle count (N)", cxxopts::value<std::size_t>()->default_value("6")},
       {"b,beta", "Inverse temperature (beta)", cxxopts::value<double>()->default_value("1.5")},
       {"l,linear-size", "Linear lattice size (L)",
        cxxopts::value<qmc::Coord>()->default_value("8")},
       {"d,dimension", "Spatial dimension (d)", cxxopts::value<std::size_t>()->default_value("1")},
       {"t,hopping", "Hopping amplitude (t)", cxxopts::value<double>()->default_value("1.0")},
       {"u,interaction", "On-site pair interaction (U)",
        cxxopts::value<double>()->default_value("2.0")},
       {"s,seed", "Random seed", cxxopts::value<std::uint64_t>()->default_value("20260717")},
       {"samples", "Measurement samples", cxxopts::value<std::size_t>()->default_value("3000")},
       {"burn-in", "Burn-in sweeps", cxxopts::value<std::size_t>()->default_value("500")},
       {"thin", "Sweeps between measurements", cxxopts::value<std::size_t>()->default_value("1")},
       {"segment-updates", "Segment updates per sweep (default N)", cxxopts::value<std::size_t>()},
       {"segment-fraction", "Fraction of beta redrawn by a segment update",
        cxxopts::value<double>()->default_value("0.35")},
       {"cycle-updates", "Cycle updates per sweep",
        cxxopts::value<std::size_t>()->default_value("1")},
       {"global-updates", "Global ideal proposals per sweep",
        cxxopts::value<std::size_t>()->default_value("0")},
       {"stitch-updates", "Random-seam stitch attempts per sweep (default max(1, N))",
        cxxopts::value<std::size_t>()},
       {"stitch-fraction", "Fraction of beta redrawn by random-seam stitches",
        cxxopts::value<double>()->default_value("0.75")},
       {"stitch-locality-radius", "Torus Chebyshev radius for local stitch partners",
        cxxopts::value<std::size_t>()->default_value("1")},
       {"stitch-global-partner-probability",
        "Probability of selecting a uniformly global stitch partner",
        cxxopts::value<double>()->default_value("0.05")},
       {"stitch-strand-counts", "Comma-separated collective stitch sizes in [2, 8]",
        cxxopts::value<std::vector<std::size_t>>()->default_value("2")},
       {"stitch-strand-weights", "Optional comma-separated weights for collective stitch sizes",
        cxxopts::value<std::vector<double>>()},
       {"o,output", "Output trace table",
        cxxopts::value<std::string>()->default_value("interacting_trace.dat")},
       {"density-momenta",
        "Semicolon-separated momentum rows with comma-separated components (for example "
        "'1,0;0,1')",
        cxxopts::value<std::string>()},
       {"density-frequency-max", "Largest bosonic Matsubara index in the inclusive range [0,n]",
        cxxopts::value<std::size_t>()},
       {"density-lags",
        "Comma-separated imaginary-time lags in the canonical interval [0,beta), mutually "
        "exclusive with --density-frequency-max",
        cxxopts::value<std::vector<double>>()},
       {"density-measurements-per-block", "Consecutive density measurements in each block",
        cxxopts::value<std::size_t>()},
       {"density-continuation-dir", "New density-continuation-v1 output directory",
        cxxopts::value<std::string>()},
       {"hopping-momenta",
        "Semicolon-separated hopping-response momentum rows with comma-separated components",
        cxxopts::value<std::string>()},
       {"hopping-frequency-max",
        "Largest hopping-response bosonic Matsubara index in the inclusive range [0,n]",
        cxxopts::value<std::size_t>()},
       {"hopping-measurements-per-block", "Consecutive hopping-response measurements in each block",
        cxxopts::value<std::size_t>()},
       {"hopping-response-dir", "New hopping-response-v1 output directory",
        cxxopts::value<std::string>()},
       {"no-trace", "Do not retain the scalar trace when writing a measurement bundle",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true")},
       {"h,help", "Print help"}});

  const auto result = options.parse(argc, argv);
  if (result.contains("help")) {
    std::cout << options.help();
    return std::nullopt;
  }
  const auto samples = result["samples"].as<std::size_t>();
  const auto thin = result["thin"].as<std::size_t>();
  if (samples == 0) {
    throw std::invalid_argument("samples must be positive");
  }
  if (thin == 0) {
    throw std::invalid_argument("thin must be positive");
  }
  std::optional<std::size_t> segment_updates;
  if (result.contains("segment-updates")) {
    segment_updates = result["segment-updates"].as<std::size_t>();
  }
  const auto particle_count = result["particles"].as<std::size_t>();
  std::size_t stitch_updates = 0;
  if (result.contains("stitch-updates")) {
    stitch_updates = result["stitch-updates"].as<std::size_t>();
  } else if (particle_count != 0) {
    stitch_updates = std::max<std::size_t>(1, particle_count);
  }
  std::vector<double> stitch_strand_weights;
  if (result.contains("stitch-strand-weights")) {
    stitch_strand_weights = result["stitch-strand-weights"].as<std::vector<double>>();
  }

  const qmc::InteractingModel model{
      .free = qmc::Model(qmc::ModelParameters{.particle_count = particle_count,
                                              .beta = result["beta"].as<double>(),
                                              .linear_size = result["linear-size"].as<qmc::Coord>(),
                                              .dimension = result["dimension"].as<std::size_t>(),
                                              .hopping = result["hopping"].as<double>()}),
      .interaction = result["interaction"].as<double>(),
  };
  model.validate();

  const bool retain_scalar_trace = !result["no-trace"].as<bool>();
  std::optional<DensityContinuationCommand> density_continuation =
      parse_density_continuation(result, model, samples);
  std::optional<HoppingResponseCommand> hopping_response =
      parse_hopping_response(result, model, samples);
  if (!retain_scalar_trace && !density_continuation.has_value() && !hopping_response.has_value()) {
    throw std::invalid_argument(
        "--no-trace requires --density-continuation-dir or --hopping-response-dir");
  }

  return CommandLine{
      .model = model,
      .seed = result["seed"].as<std::uint64_t>(),
      .samples = samples,
      .burn_in = result["burn-in"].as<std::size_t>(),
      .thin = thin,
      .sweep = {.segment_updates = segment_updates,
                .segment_fraction = result["segment-fraction"].as<double>(),
                .cycle_updates = result["cycle-updates"].as<std::size_t>(),
                .global_updates = result["global-updates"].as<std::size_t>(),
                .stitch_mixture = {}},
      .random_seam_stitch =
          {
              .updates = stitch_updates,
              .fraction = result["stitch-fraction"].as<double>(),
              .locality_radius = result["stitch-locality-radius"].as<std::size_t>(),
              .global_partner_probability =
                  result["stitch-global-partner-probability"].as<double>(),
              .mixture = {.strand_counts =
                              result["stitch-strand-counts"].as<std::vector<std::size_t>>(),
                          .strand_weights = std::move(stitch_strand_weights)},
          },
      .output = result["output"].as<std::string>(),
      .retain_scalar_trace = retain_scalar_trace,
      .density_continuation = std::move(density_continuation),
      .hopping_response = std::move(hopping_response),
  };
}

bool path_starts_with(const std::filesystem::path &path, const std::filesystem::path &prefix) {
  auto path_component = path.begin();
  for (auto prefix_component = prefix.begin(); prefix_component != prefix.end();
       ++prefix_component, ++path_component) {
    if (path_component == path.end() || *path_component != *prefix_component) {
      return false;
    }
  }
  return true;
}

void ensure_parent_directory(const std::filesystem::path &path) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

double winding_squared(const qmc::Site &winding) {
  double value = 0.0;
  for (const qmc::Coord component : winding) {
    const auto coordinate = static_cast<double>(component);
    value += coordinate * coordinate;
  }
  return value;
}

void print_acceptance(const qmc::InteractingSampler &sampler, const qmc::MoveKind move) {
  std::cout << qmc::move_name(move) << " acceptance = ";
  const auto acceptance = sampler.statistics(move).acceptance();
  if (acceptance.has_value()) {
    std::cout << *acceptance;
  } else {
    std::cout << "n/a";
  }
  std::cout << '\n';
}

struct PreparedOutputs {
  std::filesystem::path trace_path;
  std::filesystem::path density_continuation_path;
  std::filesystem::path hopping_response_path;
  std::optional<std::ofstream> trace;
};

PreparedOutputs prepare_outputs(const CommandLine &command_line) {
  PreparedOutputs output{
      .trace_path = std::filesystem::absolute(command_line.output).lexically_normal(),
      .density_continuation_path = {},
      .hopping_response_path = {},
      .trace = std::nullopt,
  };
  if (command_line.density_continuation.has_value()) {
    output.density_continuation_path =
        std::filesystem::absolute(command_line.density_continuation->output_directory)
            .lexically_normal();
    if (command_line.retain_scalar_trace &&
        (path_starts_with(output.trace_path, output.density_continuation_path) ||
         path_starts_with(output.density_continuation_path, output.trace_path))) {
      throw std::invalid_argument(
          "scalar trace and continuation bundle paths must not contain one another");
    }
    ensure_parent_directory(output.density_continuation_path);
    qmc::example::validate_density_continuation_bundle_destination(
        output.density_continuation_path);
  }
  if (command_line.hopping_response.has_value()) {
    output.hopping_response_path =
        std::filesystem::absolute(command_line.hopping_response->output_directory)
            .lexically_normal();
    if (command_line.retain_scalar_trace &&
        (path_starts_with(output.trace_path, output.hopping_response_path) ||
         path_starts_with(output.hopping_response_path, output.trace_path))) {
      throw std::invalid_argument(
          "scalar trace and hopping-response bundle paths must not contain one another");
    }
    if (command_line.density_continuation.has_value() &&
        (path_starts_with(output.density_continuation_path, output.hopping_response_path) ||
         path_starts_with(output.hopping_response_path, output.density_continuation_path))) {
      throw std::invalid_argument(
          "density and hopping-response bundle paths must not contain one another");
    }
    ensure_parent_directory(output.hopping_response_path);
    qmc::example::validate_hopping_response_bundle_destination(output.hopping_response_path);
  }
  if (command_line.retain_scalar_trace) {
    ensure_parent_directory(output.trace_path);
    output.trace.emplace(output.trace_path);
    if (!*output.trace) {
      throw std::runtime_error("failed to open output file: " + output.trace_path.string());
    }
    *output.trace << std::setprecision(16);
    *output.trace << "# sample total_energy kinetic_energy interaction_energy "
                     "double_occupancy_per_site winding_squared event_count\n";
  }
  return output;
}

struct MatsubaraDensityWorkflow {
  qmc::ContinuousMatsubaraPlan plan;
  qmc::DensityMatsubaraBlockAccumulator accumulator;
};

struct LagDensityWorkflow {
  qmc::ContinuousDensityLagPlan plan;
  qmc::DensityLagBlockAccumulator accumulator;
};

struct DensityWorkflow {
  std::filesystem::path output_directory;
  std::variant<MatsubaraDensityWorkflow, LagDensityWorkflow> backend;
};

std::optional<DensityWorkflow> make_density_workflow(const CommandLine &command_line,
                                                     const PreparedOutputs &output) {
  if (!command_line.density_continuation.has_value()) {
    return std::nullopt;
  }
  const DensityContinuationCommand &density = *command_line.density_continuation;
  if (const auto *modes = std::get_if<qmc::MatsubaraModeSet>(&density.geometry)) {
    return DensityWorkflow{
        .output_directory = output.density_continuation_path,
        .backend =
            MatsubaraDensityWorkflow{
                .plan = qmc::ContinuousMatsubaraPlan(*modes),
                .accumulator = qmc::DensityMatsubaraBlockAccumulator(
                    command_line.model.free, *modes, density.measurements_per_block),
            },
    };
  }
  const auto &lags = std::get<qmc::ImaginaryTimeLagSet>(density.geometry);
  return DensityWorkflow{
      .output_directory = output.density_continuation_path,
      .backend =
          LagDensityWorkflow{
              .plan = qmc::ContinuousDensityLagPlan(lags),
              .accumulator = qmc::DensityLagBlockAccumulator(command_line.model.free, lags,
                                                             density.measurements_per_block),
          },
  };
}

struct HoppingWorkflow {
  std::filesystem::path output_directory;
  qmc::ContinuousMatsubaraPlan plan;
  qmc::HoppingResponseBlockAccumulator accumulator;
};

std::optional<HoppingWorkflow> make_hopping_workflow(const CommandLine &command_line,
                                                     const PreparedOutputs &output) {
  if (!command_line.hopping_response.has_value()) {
    return std::nullopt;
  }
  const HoppingResponseCommand &hopping = *command_line.hopping_response;
  return HoppingWorkflow{
      .output_directory = output.hopping_response_path,
      .plan = qmc::ContinuousMatsubaraPlan(hopping.modes),
      .accumulator = qmc::HoppingResponseBlockAccumulator(command_line.model.free, hopping.modes,
                                                          hopping.measurements_per_block),
  };
}

void advance_sampler(qmc::InteractingSampler &sampler, const CommandLine &command_line) {
  if (command_line.random_seam_stitch.updates.value_or(0) != 0) {
    sampler.random_seam_stitch_sweep(command_line.random_seam_stitch);
  }
  sampler.sweep(command_line.sweep);
}

struct ScalarTotals {
  double energy = 0.0;
  double occupancy = 0.0;
  double winding = 0.0;
};

ScalarTotals measure_samples(qmc::InteractingSampler &sampler, const CommandLine &command_line,
                             PreparedOutputs &output,
                             std::optional<DensityWorkflow> &density_workflow,
                             std::optional<HoppingWorkflow> &hopping_workflow) {
  ScalarTotals totals;
  for (std::size_t sample = 0; sample < command_line.samples; ++sample) {
    for (std::size_t thin = 0; thin < command_line.thin; ++thin) {
      advance_sampler(sampler, command_line);
    }
    const qmc::InteractingObservables value = sampler.observables();
    const double winding = winding_squared(value.winding);
    if (output.trace.has_value()) {
      *output.trace << sample << ' ' << value.total_energy << ' ' << value.kinetic_energy << ' '
                    << value.interaction_energy << ' ' << value.double_occupancy_per_site << ' '
                    << winding << ' ' << value.event_count << '\n';
    }
    if (density_workflow.has_value() || hopping_workflow.has_value()) {
      const qmc::ContinuousMeasurementContext context(sampler.state());
      if (density_workflow.has_value()) {
        std::visit(
            [&context](auto &backend) {
              using Backend = std::remove_cvref_t<decltype(backend)>;
              if constexpr (std::is_same_v<Backend, MatsubaraDensityWorkflow>) {
                backend.accumulator.observe(qmc::continuous_particle_modes(context, backend.plan));
              } else {
                backend.accumulator.observe(
                    qmc::continuous_density_lag_values(context, backend.plan));
              }
            },
            density_workflow->backend);
      }
      if (hopping_workflow.has_value()) {
        hopping_workflow->accumulator.observe(
            qmc::continuous_particle_modes(context, hopping_workflow->plan));
      }
    }
    totals.energy += value.total_energy;
    totals.occupancy += value.double_occupancy_per_site;
    totals.winding += winding;
  }
  return totals;
}

void finish_trace(PreparedOutputs &output) {
  if (!output.trace.has_value()) {
    return;
  }
  output.trace->close();
  if (!*output.trace) {
    throw std::runtime_error("failed to write output file: " + output.trace_path.string());
  }
}

double largest_standard_error(const qmc::DensityMatsubaraBlockSeries &series) {
  double largest = 0.0;
  for (std::size_t momentum = 0; momentum < series.modes().momentum_count(); ++momentum) {
    for (std::size_t frequency = 0; frequency < series.modes().frequency_count(); ++frequency) {
      largest = std::max(largest, series.standard_error(frequency, momentum));
    }
  }
  return largest;
}

double largest_standard_error(const qmc::DensityLagBlockSeries &series) {
  double largest = 0.0;
  for (std::size_t momentum = 0; momentum < series.lags().momentum_count(); ++momentum) {
    for (std::size_t lag = 0; lag < series.lags().lag_count(); ++lag) {
      largest = std::max(largest, series.standard_error(lag, momentum));
    }
  }
  return largest;
}

void publish_density_workflow(DensityWorkflow &workflow, const CommandLine &command_line) {
  const qmc::example::DensityContinuationRunProvenance provenance{
      .model = command_line.model,
      .seed = command_line.seed,
      .burn_in_sweeps = command_line.burn_in,
      .thinning_sweeps = command_line.thin,
      .sweep = command_line.sweep,
      .random_seam_stitch = command_line.random_seam_stitch,
      .scalar_trace_retained = command_line.retain_scalar_trace,
      .program = "qmc_interacting_demo",
      .program_version = std::string(qmc::kVersion),
  };
  std::visit(
      [&workflow, &provenance](auto &backend) {
        const auto series = backend.accumulator.finish();
        qmc::example::write_density_continuation_bundle(workflow.output_directory, series,
                                                        provenance);
        std::cout << "continuation bundle written to " << workflow.output_directory << '\n';
        std::cout << "density blocks = " << series.block_count() << '\n';
        std::cout << "largest density standard error = " << largest_standard_error(series) << '\n';
      },
      workflow.backend);
}

double largest_flux_response_standard_error(const qmc::HoppingResponseBlockSeries &series) {
  double largest = 0.0;
  for (std::size_t frequency = 0; frequency < series.modes().frequency_count(); ++frequency) {
    for (std::size_t momentum = 0; momentum < series.modes().momentum_count(); ++momentum) {
      for (std::size_t left = 0; left < series.model().dimension(); ++left) {
        for (std::size_t right = 0; right < series.model().dimension(); ++right) {
          largest = std::max(
              largest, series.flux_response_standard_error(frequency, momentum, left, right,
                                                           qmc::HoppingResponseComponent::Real));
          largest = std::max(largest, series.flux_response_standard_error(
                                          frequency, momentum, left, right,
                                          qmc::HoppingResponseComponent::Imaginary));
        }
      }
    }
  }
  return largest;
}

void publish_hopping_workflow(HoppingWorkflow &workflow, const CommandLine &command_line) {
  const qmc::example::HoppingResponseRunProvenance provenance{
      .model = command_line.model,
      .seed = command_line.seed,
      .burn_in_sweeps = command_line.burn_in,
      .thinning_sweeps = command_line.thin,
      .sweep = command_line.sweep,
      .random_seam_stitch = command_line.random_seam_stitch,
      .scalar_trace_retained = command_line.retain_scalar_trace,
      .program = "qmc_interacting_demo",
      .program_version = std::string(qmc::kVersion),
  };
  const qmc::HoppingResponseBlockSeries series = workflow.accumulator.finish();
  qmc::example::write_hopping_response_bundle(workflow.output_directory, series, provenance);
  std::cout << "hopping-response bundle written to " << workflow.output_directory << '\n';
  std::cout << "hopping-response blocks = " << series.block_count() << '\n';
  std::cout << "largest hopping-response standard error = "
            << largest_flux_response_standard_error(series) << '\n';
}

void print_run_summary(const qmc::InteractingSampler &sampler, const CommandLine &command_line,
                       const PreparedOutputs &output, const ScalarTotals totals) {
  const auto samples = static_cast<double>(command_line.samples);
  std::cout << std::setprecision(10);
  if (command_line.retain_scalar_trace) {
    std::cout << "trace written to " << output.trace_path << '\n';
  }
  std::cout << "<E> = " << totals.energy / samples << '\n';
  std::cout << "<D>/site = " << totals.occupancy / samples << '\n';
  std::cout << "<W^2> = " << totals.winding / samples << '\n';
  print_acceptance(sampler, qmc::MoveKind::SegmentMove);
  print_acceptance(sampler, qmc::MoveKind::CycleMove);
  print_acceptance(sampler, qmc::MoveKind::StitchMove);
  const auto &stitch = sampler.statistics(qmc::MoveKind::StitchMove);
  std::cout << "stitch topology changes/attempt = ";
  if (const auto rate = stitch.topology_change_rate(); rate.has_value()) {
    std::cout << *rate;
  } else {
    std::cout << "n/a";
  }
  std::cout << '\n';
  std::cout << "stitch successor changes/attempt = ";
  if (const auto rate = stitch.successor_changes_per_attempt(); rate.has_value()) {
    std::cout << *rate;
  } else {
    std::cout << "n/a";
  }
  std::cout << '\n';
  print_acceptance(sampler, qmc::MoveKind::TimeShiftMove);
  print_acceptance(sampler, qmc::MoveKind::GlobalMove);
}

void execute_demo(const CommandLine &command_line) {
  PreparedOutputs output = prepare_outputs(command_line);
  std::optional<DensityWorkflow> density_workflow = make_density_workflow(command_line, output);
  std::optional<HoppingWorkflow> hopping_workflow = make_hopping_workflow(command_line, output);
  qmc::InteractingSampler sampler(command_line.model, command_line.seed);
  for (std::size_t sweep = 0; sweep < command_line.burn_in; ++sweep) {
    advance_sampler(sampler, command_line);
  }
  const ScalarTotals totals =
      measure_samples(sampler, command_line, output, density_workflow, hopping_workflow);
  finish_trace(output);
  if (density_workflow.has_value()) {
    publish_density_workflow(*density_workflow, command_line);
  }
  if (hopping_workflow.has_value()) {
    publish_hopping_workflow(*hopping_workflow, command_line);
  }
  print_run_summary(sampler, command_line, output, totals);
}

} // namespace

int main(const int argc, char **argv) {
  try {
    const auto command_line = parse_command_line(argc, argv);
    if (!command_line.has_value()) {
      return 0;
    }
    execute_demo(*command_line);
  } catch (const cxxopts::exceptions::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    std::cerr << "Try '" << argv[0] << " --help' for usage.\n";
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
