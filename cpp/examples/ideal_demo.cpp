#include "qmc/ideal.hpp"

#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>

namespace {

struct CommandLine {
  qmc::Model model;
  std::size_t time_links;
  std::uint64_t seed;
};

std::optional<CommandLine> parse_command_line(const int argc, char **argv) {
  cxxopts::Options options(argv[0], "Sample an ideal-boson world-line configuration");
  options.positional_help("[N] [beta] [M] [L] [d] [t] [seed]").show_positional_help();
  options.add_options(
      "",
      {
          {"n,particles", "Particle count (N)", cxxopts::value<std::size_t>()->default_value("4")},
          {"b,beta", "Inverse temperature (beta)", cxxopts::value<double>()->default_value("1.0")},
          {"m,time-links", "Retained time links per beta (M)",
           cxxopts::value<std::size_t>()->default_value("64")},
          {"l,linear-size", "Linear lattice size (L)",
           cxxopts::value<qmc::Coord>()->default_value("12")},
          {"d,dimension", "Spatial dimension (d)",
           cxxopts::value<std::size_t>()->default_value("1")},
          {"t,hopping", "Hopping amplitude (t)", cxxopts::value<double>()->default_value("1.0")},
          {"s,seed", "Random seed", cxxopts::value<std::uint64_t>()->default_value("2026")},
          {"h,help", "Print help"},
      });
  options.parse_positional(
      {"particles", "beta", "time-links", "linear-size", "dimension", "hopping", "seed"});

  const auto result = options.parse(argc, argv);
  if (result.contains("help")) {
    std::cout << options.help();
    return std::nullopt;
  }

  return CommandLine{
      .model =
          {
              .particle_count = result["particles"].as<std::size_t>(),
              .beta = result["beta"].as<double>(),
              .linear_size = result["linear-size"].as<qmc::Coord>(),
              .dimension = result["dimension"].as<std::size_t>(),
              .hopping = result["hopping"].as<double>(),
          },
      .time_links = result["time-links"].as<std::size_t>(),
      .seed = result["seed"].as<std::uint64_t>(),
  };
}

void print_site(const qmc::Site &site) {
  std::cout << '[';
  for (std::size_t axis = 0; axis < site.size(); ++axis) {
    if (axis != 0) {
      std::cout << ", ";
    }
    std::cout << site[axis];
  }
  std::cout << ']';
}

} // namespace

int main(const int argc, char **argv) {
  try {
    const auto command_line = parse_command_line(argc, argv);
    if (!command_line.has_value()) {
      return 0;
    }

    qmc::Random random(command_line->seed);
    const auto configuration = qmc::sample_ideal_boson_configuration(
        command_line->model, command_line->time_links, random);

    std::cout << std::setprecision(12) << "log Z_N = " << configuration.log_ZN << '\n';
    std::cout << "permutation = [";
    for (std::size_t index = 0; index < configuration.permutation.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << configuration.permutation[index];
    }
    std::cout << "]\n";

    for (std::size_t index = 0; index < configuration.cycles.size(); ++index) {
      const auto &cycle = configuration.cycles[index];
      std::cout << "cycle " << index << ": labels=[";
      for (std::size_t label_index = 0; label_index < cycle.labels.size(); ++label_index) {
        if (label_index != 0) {
          std::cout << ", ";
        }
        std::cout << cycle.labels[label_index];
      }
      std::cout << "], base=";
      print_site(cycle.base_point);
      std::cout << ", winding=";
      print_site(cycle.winding);
      std::cout << '\n';
    }
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
