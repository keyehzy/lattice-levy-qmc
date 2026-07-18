#include "qmc/ideal.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::size_t parse_size(const char *text, const std::string_view name) {
  const auto parsed = std::stoull(text);
  if (parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::out_of_range(std::string(name) + " exceeds size_t");
  }
  return static_cast<std::size_t>(parsed);
}

qmc::Coord parse_coord(const char *text, const std::string_view name) {
  const auto parsed = std::stoll(text);
  if (parsed < std::numeric_limits<qmc::Coord>::min() ||
      parsed > std::numeric_limits<qmc::Coord>::max()) {
    throw std::out_of_range(std::string(name) + " exceeds Coord");
  }
  return static_cast<qmc::Coord>(parsed);
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
    if (argc > 8) {
      std::cerr << "usage: " << argv[0] << " [N beta M L d t seed]\n";
      return 2;
    }

    qmc::Model model{
        .particle_count = argc > 1 ? parse_size(argv[1], "N") : 4,
        .beta = argc > 2 ? std::stod(argv[2]) : 1.0,
        .linear_size = argc > 4 ? parse_coord(argv[4], "L") : 12,
        .dimension = argc > 5 ? parse_size(argv[5], "d") : 1,
        .hopping = argc > 6 ? std::stod(argv[6]) : 1.0,
    };
    const std::size_t time_links = argc > 3 ? parse_size(argv[3], "M") : 64;
    const std::uint64_t seed = argc > 7 ? std::stoull(argv[7]) : 2026;

    qmc::Random random(seed);
    const auto configuration = qmc::sample_ideal_boson_configuration(model, time_links, random);

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
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
