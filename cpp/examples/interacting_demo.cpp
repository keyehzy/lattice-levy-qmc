#include "qmc/interacting.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct CommandLine {
  qmc::InteractingModel model;
  std::uint64_t seed;
  std::size_t samples;
  std::size_t burn_in;
  std::size_t thin;
  qmc::SweepOptions sweep;
  std::size_t stitch_updates;
  double stitch_fraction;
  std::size_t stitch_locality_radius;
  double stitch_global_partner_probability;
  qmc::StitchMixture stitch_mixture;
  std::filesystem::path output;
};

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

  return CommandLine{
      .model =
          {
              .free = {.particle_count = particle_count,
                       .beta = result["beta"].as<double>(),
                       .linear_size = result["linear-size"].as<qmc::Coord>(),
                       .dimension = result["dimension"].as<std::size_t>(),
                       .hopping = result["hopping"].as<double>()},
              .interaction = result["interaction"].as<double>(),
          },
      .seed = result["seed"].as<std::uint64_t>(),
      .samples = samples,
      .burn_in = result["burn-in"].as<std::size_t>(),
      .thin = thin,
      .sweep = {.segment_updates = segment_updates,
                .segment_fraction = result["segment-fraction"].as<double>(),
                .cycle_updates = result["cycle-updates"].as<std::size_t>(),
                .global_updates = result["global-updates"].as<std::size_t>(),
                .stitch_mixture = {}},
      .stitch_updates = stitch_updates,
      .stitch_fraction = result["stitch-fraction"].as<double>(),
      .stitch_locality_radius = result["stitch-locality-radius"].as<std::size_t>(),
      .stitch_global_partner_probability = result["stitch-global-partner-probability"].as<double>(),
      .stitch_mixture = {.strand_counts =
                             result["stitch-strand-counts"].as<std::vector<std::size_t>>(),
                         .strand_weights = std::move(stitch_strand_weights)},
      .output = result["output"].as<std::string>(),
  };
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

} // namespace

int main(const int argc, char **argv) {
  try {
    const auto command_line = parse_command_line(argc, argv);
    if (!command_line.has_value()) {
      return 0;
    }
    command_line->model.validate();
    qmc::InteractingSampler sampler(command_line->model, command_line->seed);
    const auto advance = [&sampler, &command_line] {
      if (command_line->stitch_updates != 0) {
        sampler.random_seam_stitch_sweep(
            command_line->stitch_updates, command_line->stitch_fraction,
            command_line->stitch_locality_radius, command_line->stitch_global_partner_probability,
            command_line->stitch_mixture);
      }
      sampler.sweep(command_line->sweep);
    };
    for (std::size_t sweep = 0; sweep < command_line->burn_in; ++sweep) {
      advance();
    }

    const auto absolute_output = std::filesystem::absolute(command_line->output);
    if (!absolute_output.parent_path().empty()) {
      std::filesystem::create_directories(absolute_output.parent_path());
    }
    std::ofstream output(absolute_output);
    if (!output) {
      throw std::runtime_error("failed to open output file: " + absolute_output.string());
    }
    output << std::setprecision(16);
    output << "# sample total_energy kinetic_energy interaction_energy "
              "double_occupancy_per_site winding_squared event_count\n";

    double energy_sum = 0.0;
    double occupancy_sum = 0.0;
    double winding_sum = 0.0;
    for (std::size_t sample = 0; sample < command_line->samples; ++sample) {
      for (std::size_t thin = 0; thin < command_line->thin; ++thin) {
        advance();
      }
      const auto value = sampler.observables();
      const double winding = winding_squared(value.winding);
      output << sample << ' ' << value.total_energy << ' ' << value.kinetic_energy << ' '
             << value.interaction_energy << ' ' << value.double_occupancy_per_site << ' ' << winding
             << ' ' << value.event_count << '\n';
      energy_sum += value.total_energy;
      occupancy_sum += value.double_occupancy_per_site;
      winding_sum += winding;
    }

    const auto samples = static_cast<double>(command_line->samples);
    std::cout << std::setprecision(10);
    std::cout << "trace written to " << absolute_output << '\n';
    std::cout << "<E> = " << energy_sum / samples << '\n';
    std::cout << "<D>/site = " << occupancy_sum / samples << '\n';
    std::cout << "<W^2> = " << winding_sum / samples << '\n';
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
