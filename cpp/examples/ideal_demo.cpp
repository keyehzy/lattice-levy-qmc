#include "qmc/ideal.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cxxopts.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CommandLine {
  qmc::Model model;
  std::size_t time_links;
  std::uint64_t seed;
  std::size_t samples;
  std::filesystem::path output_directory;
  bool plot;
  std::string gnuplot;
};

std::optional<CommandLine> parse_command_line(const int argc, char **argv) {
  cxxopts::Options options(argv[0], "Measure an exact canonical ideal-boson world-line ensemble");
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
           cxxopts::value<std::size_t>()->default_value("2")},
          {"t,hopping", "Hopping amplitude (t)", cxxopts::value<double>()->default_value("1.0")},
          {"s,seed", "Random seed", cxxopts::value<std::uint64_t>()->default_value("2026")},
          {"samples", "Independent configurations to average",
           cxxopts::value<std::size_t>()->default_value("200")},
          {"output-dir", "Directory for data, script, and plots",
           cxxopts::value<std::string>()->default_value("ideal_observables")},
          {"plot", "Run the generated gnuplot script",
           cxxopts::value<bool>()->default_value("false")->implicit_value("true")},
          {"gnuplot", "gnuplot executable",
           cxxopts::value<std::string>()->default_value("gnuplot")},
          {"h,help", "Print help"},
      });
  options.parse_positional(
      {"particles", "beta", "time-links", "linear-size", "dimension", "hopping", "seed"});

  const auto result = options.parse(argc, argv);
  if (result.contains("help")) {
    std::cout << options.help();
    return std::nullopt;
  }
  const auto samples = result["samples"].as<std::size_t>();
  if (samples == 0) {
    throw std::invalid_argument("samples must be positive");
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
      .samples = samples,
      .output_directory = result["output-dir"].as<std::string>(),
      .plot = result["plot"].as<bool>(),
      .gnuplot = result["gnuplot"].as<std::string>(),
  };
}

std::ofstream output_file(const std::filesystem::path &path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open output file: " + path.string());
  }
  output << std::setprecision(16);
  return output;
}

void add_values(std::vector<double> &destination, const std::span<const double> source) {
  if (destination.size() != source.size()) {
    throw std::logic_error("observable accumulator shape mismatch");
  }
  for (std::size_t index = 0; index < destination.size(); ++index) {
    destination[index] += source[index];
  }
}

void divide_values(std::vector<double> &values, const double divisor) {
  for (double &value : values) {
    value /= divisor;
  }
}

double winding_squared(const qmc::Site &winding) {
  double result = 0.0;
  for (const qmc::Coord component : winding) {
    const auto value = static_cast<double>(component);
    result += value * value;
  }
  return result;
}

std::size_t axis_cut_flat_index(const std::size_t first_component, const qmc::Model &model) {
  const auto linear_size = static_cast<std::size_t>(model.linear_size);
  if (first_component >= linear_size) {
    throw std::out_of_range("axis-cut component is outside the torus");
  }
  return first_component;
}

std::size_t farthest_flat_index(const qmc::Model &model) {
  const auto component = static_cast<std::size_t>(model.linear_size / 2);
  const qmc::TorusLayout layout(model.linear_size, model.dimension);
  return layout.encode(std::vector<std::size_t>(model.dimension, component)).value();
}

void write_index_components(std::ostream &output, std::span<const std::size_t> components) {
  for (const std::size_t component : components) {
    output << ' ' << component;
  }
}

void write_site_components(std::ostream &output, const qmc::Site &site) {
  for (const qmc::Coord component : site) {
    output << ' ' << component;
  }
}

std::string gnuplot_quote(const std::filesystem::path &path) {
  std::string result;
  for (const char character : path.string()) {
    if (character == '\\' || character == '\'') {
      result.push_back('\\');
    }
    result.push_back(character);
  }
  return "'" + result + "'";
}

std::string shell_quote(const std::string &value) {
  std::string result = "'";
  for (const char character : value) {
    if (character == '\'') {
      result += "'\\''";
    } else {
      result.push_back(character);
    }
  }
  result.push_back('\'');
  return result;
}

struct CycleGeometrySample {
  std::size_t sample = 0;
  std::size_t length = 0;
  double total_winding_squared = 0.0;
  double cycle_winding_squared = 0.0;
  double radius_of_gyration_squared = 0.0;
  double maximum_radius_squared = 0.0;
  qmc::Site cycle_winding;
};

struct SampleAverages {
  std::vector<double> cycle_count;
  std::vector<double> cycle_particles;
  std::vector<double> cycle_winding_squared;
  std::vector<double> cycle_radius_of_gyration_squared;
  std::vector<double> cycle_maximum_radius_squared;
  std::vector<double> cycle_occurrences;
  std::vector<double> longest_cycle_probability;
  std::vector<double> site_density;
  std::vector<double> pair_correlation;
  std::vector<double> structure_factor;
  std::vector<double> onsite_probability;
  std::vector<double> connected_density;
  std::vector<double> mean_square_displacement;
  std::vector<double> return_probability;
  std::vector<double> displacement_probability;
  std::vector<double> winding_second_moment;
  std::vector<double> winding_fourth_moment;
  std::vector<double> winding_squared_trace;
  std::map<qmc::Site, std::size_t> winding_histogram;
  std::map<double, std::size_t> winding_squared_histogram;
  std::map<std::pair<double, std::size_t>, std::size_t> winding_conditioned_cycles;
  std::vector<CycleGeometrySample> cycle_geometry_samples;
  double mean_occupation_squared = 0.0;
  double mean_factorial_occupation = 0.0;
  double nonzero_winding_probability = 0.0;
  double macroscopic_cycle_fraction = 0.0;
};

SampleAverages sample_ensemble(const CommandLine &command_line,
                               const qmc::CanonicalEnsemble &ensemble, qmc::Random &random) {
  const auto &model = command_line.model;
  const auto volume = model.volume();
  SampleAverages averages{
      .cycle_count = std::vector<double>(model.particle_count + 1),
      .cycle_particles = std::vector<double>(model.particle_count + 1),
      .cycle_winding_squared = std::vector<double>(model.particle_count + 1),
      .cycle_radius_of_gyration_squared = std::vector<double>(model.particle_count + 1),
      .cycle_maximum_radius_squared = std::vector<double>(model.particle_count + 1),
      .cycle_occurrences = std::vector<double>(model.particle_count + 1),
      .longest_cycle_probability = std::vector<double>(model.particle_count + 1),
      .site_density = std::vector<double>(volume),
      .pair_correlation = std::vector<double>(volume),
      .structure_factor = std::vector<double>(volume),
      .onsite_probability = std::vector<double>(model.particle_count + 1),
      .connected_density = std::vector<double>(command_line.time_links * volume),
      .mean_square_displacement = std::vector<double>(command_line.time_links),
      .return_probability = std::vector<double>(command_line.time_links),
      .displacement_probability = std::vector<double>(command_line.time_links * volume),
      .winding_second_moment = std::vector<double>(model.dimension),
      .winding_fourth_moment = std::vector<double>(model.dimension),
      .winding_squared_trace = {},
      .winding_histogram = {},
      .winding_squared_histogram = {},
      .winding_conditioned_cycles = {},
      .cycle_geometry_samples = {},
      .mean_occupation_squared = 0.0,
      .mean_factorial_occupation = 0.0,
      .nonzero_winding_probability = 0.0,
      .macroscopic_cycle_fraction = 0.0,
  };
  averages.winding_squared_trace.reserve(command_line.samples);
  const std::size_t macroscopic_threshold =
      model.particle_count == 0 ? 1 : (model.particle_count + 1) / 2;

  for (std::size_t sample = 0; sample < command_line.samples; ++sample) {
    const auto configuration =
        qmc::sample_ideal_boson_configuration(ensemble, command_line.time_links, random);
    const auto cycle_histogram = qmc::sampled_cycle_histogram(configuration);
    for (std::size_t length = 1; length <= model.particle_count; ++length) {
      averages.cycle_count[length] += static_cast<double>(cycle_histogram[length]);
      averages.cycle_particles[length] += static_cast<double>(length * cycle_histogram[length]);
    }
    ++averages.longest_cycle_probability[qmc::longest_cycle_length(configuration)];

    const qmc::Site winding = qmc::total_winding(configuration);
    const double total_winding_squared = winding_squared(winding);
    averages.winding_squared_trace.push_back(total_winding_squared);
    ++averages.winding_histogram[winding];
    ++averages.winding_squared_histogram[total_winding_squared];
    if (total_winding_squared > 0.0) {
      averages.nonzero_winding_probability += 1.0;
    }
    for (std::size_t axis = 0; axis < model.dimension; ++axis) {
      const auto component = static_cast<double>(winding[axis]);
      averages.winding_second_moment[axis] += component * component;
      averages.winding_fourth_moment[axis] += component * component * component * component;
    }

    const auto cycle_geometry = qmc::retained_cycle_geometry(configuration);
    std::size_t macroscopic_particles = 0;
    for (const auto &geometry : cycle_geometry) {
      const auto length = geometry.length;
      const double cycle_winding_value = winding_squared(geometry.winding);
      averages.cycle_winding_squared[length] += cycle_winding_value;
      averages.cycle_radius_of_gyration_squared[length] += geometry.radius_of_gyration_squared;
      averages.cycle_maximum_radius_squared[length] += geometry.maximum_radius_squared;
      averages.cycle_occurrences[length] += 1.0;
      ++averages.winding_conditioned_cycles[{total_winding_squared, length}];
      averages.cycle_geometry_samples.push_back(CycleGeometrySample{
          .sample = sample,
          .length = length,
          .total_winding_squared = total_winding_squared,
          .cycle_winding_squared = cycle_winding_value,
          .radius_of_gyration_squared = geometry.radius_of_gyration_squared,
          .maximum_radius_squared = geometry.maximum_radius_squared,
          .cycle_winding = geometry.winding,
      });
      if (length >= macroscopic_threshold) {
        macroscopic_particles += length;
      }
    }
    if (model.particle_count > 0) {
      averages.macroscopic_cycle_fraction +=
          static_cast<double>(macroscopic_particles) / static_cast<double>(model.particle_count);
    }

    const auto equal_time = qmc::equal_time_observables(configuration);
    add_values(averages.site_density, equal_time.site_density);
    add_values(averages.pair_correlation, equal_time.pair_correlation);
    add_values(averages.structure_factor, equal_time.static_structure_factor);
    add_values(averages.onsite_probability, equal_time.onsite_occupation_probability);
    averages.mean_occupation_squared += equal_time.mean_occupation_squared;
    averages.mean_factorial_occupation += equal_time.mean_factorial_occupation;

    const auto imaginary_time = qmc::retained_density_correlations(configuration);
    add_values(averages.connected_density, imaginary_time.connected_density());
    const auto retained_geometry = qmc::retained_geometry_observables(configuration);
    add_values(averages.mean_square_displacement, retained_geometry.mean_square_displacement);
    add_values(averages.return_probability, retained_geometry.return_probability);
    add_values(averages.displacement_probability, retained_geometry.displacement_probability);
  }

  const auto sample_count = static_cast<double>(command_line.samples);
  divide_values(averages.cycle_count, sample_count);
  divide_values(averages.cycle_particles, sample_count);
  divide_values(averages.longest_cycle_probability, sample_count);
  divide_values(averages.site_density, sample_count);
  divide_values(averages.pair_correlation, sample_count);
  divide_values(averages.structure_factor, sample_count);
  divide_values(averages.onsite_probability, sample_count);
  divide_values(averages.connected_density, sample_count);
  divide_values(averages.mean_square_displacement, sample_count);
  divide_values(averages.return_probability, sample_count);
  divide_values(averages.displacement_probability, sample_count);
  divide_values(averages.winding_second_moment, sample_count);
  divide_values(averages.winding_fourth_moment, sample_count);
  averages.mean_occupation_squared /= sample_count;
  averages.mean_factorial_occupation /= sample_count;
  averages.nonzero_winding_probability /= sample_count;
  averages.macroscopic_cycle_fraction /= sample_count;
  for (std::size_t length = 1; length <= model.particle_count; ++length) {
    if (averages.cycle_occurrences[length] > 0.0) {
      averages.cycle_winding_squared[length] /= averages.cycle_occurrences[length];
      averages.cycle_radius_of_gyration_squared[length] /= averages.cycle_occurrences[length];
      averages.cycle_maximum_radius_squared[length] /= averages.cycle_occurrences[length];
    }
  }
  return averages;
}

void write_thermodynamics(const std::filesystem::path &directory, const qmc::Model &model,
                          const std::optional<qmc::CanonicalThermodynamics> &thermodynamics) {
  auto output = output_file(directory / "thermodynamics.dat");
  output << "# N F E C S mu_addition\n";
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t particles = 0; particles <= model.particle_count; ++particles) {
    if (thermodynamics.has_value()) {
      output << particles << ' ' << thermodynamics->free_energy[particles] << ' '
             << thermodynamics->energy[particles] << ' ' << thermodynamics->heat_capacity[particles]
             << ' ' << thermodynamics->entropy[particles] << ' '
             << thermodynamics->addition_chemical_potential[particles] << '\n';
    } else {
      output << particles << ' ' << nan << ' ' << nan << ' ' << nan << ' ' << nan << ' ' << nan
             << '\n';
    }
  }
}

void write_momentum(const std::filesystem::path &directory, const qmc::Model &model,
                    const qmc::MomentumDistribution &momentum) {
  auto output = output_file(directory / "momentum_distribution.dat");
  output << "# flat k[0..d) q[0..d) epsilon occupation variance\n";
  auto cut = output_file(directory / "momentum_cut.dat");
  cut << "# k0 q0 occupation variance\n";
  for (std::size_t flat = 0; flat < momentum.modes.size(); ++flat) {
    const auto &mode = momentum.modes[flat];
    output << flat;
    write_index_components(output, mode.indices);
    for (const double component : mode.wavevector) {
      output << ' ' << component;
    }
    output << ' ' << mode.energy << ' ' << mode.occupation << ' ' << mode.occupation_variance
           << '\n';
  }
  const auto linear_size = static_cast<std::size_t>(model.linear_size);
  for (std::size_t component = 0; component < linear_size; ++component) {
    const auto &mode = momentum.modes[axis_cut_flat_index(component, model)];
    cut << component << ' ' << mode.wavevector.front() << ' ' << mode.occupation << ' '
        << mode.occupation_variance << '\n';
  }
}

void write_one_body(const std::filesystem::path &directory, const qmc::Model &model,
                    const std::vector<qmc::OneBodyDensityPoint> &density_matrix) {
  auto output = output_file(directory / "one_body_density_matrix.dat");
  output << "# flat displacement[0..d) g1\n";
  auto cut = output_file(directory / "one_body_cut.dat");
  cut << "# r0 g1\n";
  for (std::size_t flat = 0; flat < density_matrix.size(); ++flat) {
    output << flat;
    write_site_components(output, density_matrix[flat].displacement);
    output << ' ' << density_matrix[flat].value << '\n';
  }
  const auto linear_size = static_cast<std::size_t>(model.linear_size);
  for (std::size_t component = 0; component < linear_size; ++component) {
    const auto &point = density_matrix[axis_cut_flat_index(component, model)];
    cut << component << ' ' << point.value << '\n';
  }
}

void write_cycles(const std::filesystem::path &directory, const qmc::Model &model,
                  const qmc::ExactCycleStatistics &exact, const SampleAverages &sampled) {
  auto output = output_file(directory / "cycle_statistics.dat");
  output << "# l exact_m exact_particles particle_probability sampled_m sampled_particles "
            "sampled_cycle_winding2 longest_probability radius_gyration2 maximum_radius2\n";
  if (model.particle_count == 0) {
    output << "0 0 0 0 0 0 0 1 0 0\n";
  }
  for (std::size_t length = 1; length <= model.particle_count; ++length) {
    output << length << ' ' << exact.expected_cycle_count[length] << ' '
           << exact.expected_particles[length] << ' ' << exact.particle_probability[length] << ' '
           << sampled.cycle_count[length] << ' ' << sampled.cycle_particles[length] << ' '
           << sampled.cycle_winding_squared[length] << ' '
           << sampled.longest_cycle_probability[length] << ' '
           << sampled.cycle_radius_of_gyration_squared[length] << ' '
           << sampled.cycle_maximum_radius_squared[length] << '\n';
  }

  auto geometry = output_file(directory / "cycle_geometry_samples.dat");
  geometry << "# sample length total_W2 cycle_W2 radius_gyration2 maximum_radius2 cycle_W[0..d)\n";
  for (const CycleGeometrySample &record : sampled.cycle_geometry_samples) {
    geometry << record.sample << ' ' << record.length << ' ' << record.total_winding_squared << ' '
             << record.cycle_winding_squared << ' ' << record.radius_of_gyration_squared << ' '
             << record.maximum_radius_squared;
    write_site_components(geometry, record.cycle_winding);
    geometry << '\n';
  }

  auto conditioned = output_file(directory / "winding_conditioned_cycles.dat");
  conditioned << "# total_W2 length mean_cycle_count_conditioned_on_W2\n";
  for (const auto &[key, count] : sampled.winding_conditioned_cycles) {
    const auto histogram = sampled.winding_squared_histogram.find(key.first);
    if (histogram == sampled.winding_squared_histogram.end() || histogram->second == 0) {
      throw std::logic_error("winding-conditioned cycle denominator is missing");
    }
    conditioned << key.first << ' ' << key.second << ' '
                << static_cast<double>(count) / static_cast<double>(histogram->second) << '\n';
  }
}

void write_winding(const std::filesystem::path &directory, const CommandLine &command_line,
                   const std::vector<double> &twist_curvature, const SampleAverages &sampled) {
  auto trace = output_file(directory / "winding_samples.dat");
  trace << "# sample W2\n";
  for (std::size_t sample = 0; sample < sampled.winding_squared_trace.size(); ++sample) {
    trace << sample << ' ' << sampled.winding_squared_trace[sample] << '\n';
  }
  auto histogram = output_file(directory / "winding_histogram.dat");
  histogram << "# W[0..d) probability\n";
  for (const auto &[winding, count] : sampled.winding_histogram) {
    for (const qmc::Coord component : winding) {
      histogram << component << ' ';
    }
    histogram << static_cast<double>(count) / static_cast<double>(command_line.samples) << '\n';
  }
  auto squared_histogram = output_file(directory / "winding_squared_histogram.dat");
  squared_histogram << "# W2 probability\n";
  for (const auto &[value, count] : sampled.winding_squared_histogram) {
    squared_histogram << value << ' '
                      << static_cast<double>(count) / static_cast<double>(command_line.samples)
                      << '\n';
  }

  auto moments = output_file(directory / "winding_moments.dat");
  moments << "# axis W2 fourth_cumulant sampled_free_energy_curvature "
             "exact_free_energy_curvature\n";
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t axis = 0; axis < command_line.model.dimension; ++axis) {
    const double fourth_cumulant =
        sampled.winding_fourth_moment[axis] -
        (3.0 * sampled.winding_second_moment[axis] * sampled.winding_second_moment[axis]);
    const double sampled_curvature =
        command_line.model.beta > 0.0
            ? sampled.winding_second_moment[axis] / command_line.model.beta
            : nan;
    const double exact_curvature = command_line.model.beta > 0.0 ? twist_curvature[axis] : nan;
    moments << axis << ' ' << sampled.winding_second_moment[axis] << ' ' << fourth_cumulant << ' '
            << sampled_curvature << ' ' << exact_curvature << '\n';
  }
}

void write_density(const std::filesystem::path &directory, const qmc::Model &model,
                   const qmc::MomentumDistribution &momentum, const SampleAverages &sampled) {
  auto density = output_file(directory / "site_density.dat");
  density << "# flat site_density\n";
  for (std::size_t flat = 0; flat < sampled.site_density.size(); ++flat) {
    density << flat << ' ' << sampled.site_density[flat] << '\n';
  }
  auto pair = output_file(directory / "pair_correlation.dat");
  pair << "# flat displacement_index g2\n";
  auto pair_cut = output_file(directory / "pair_correlation_cut.dat");
  pair_cut << "# r0 g2\n";
  for (std::size_t flat = 0; flat < sampled.pair_correlation.size(); ++flat) {
    pair << flat;
    write_site_components(
        pair, qmc::Site(momentum.modes[flat].indices.begin(), momentum.modes[flat].indices.end()));
    pair << ' ' << sampled.pair_correlation[flat] << '\n';
  }
  const auto linear_size = static_cast<std::size_t>(model.linear_size);
  for (std::size_t component = 0; component < linear_size; ++component) {
    pair_cut << component << ' ' << sampled.pair_correlation[axis_cut_flat_index(component, model)]
             << '\n';
  }
  auto structure = output_file(directory / "static_structure_factor.dat");
  structure << "# flat k[0..d) S(q)\n";
  auto structure_cut = output_file(directory / "structure_factor_cut.dat");
  structure_cut << "# k0 q0 S(q)\n";
  for (std::size_t flat = 0; flat < sampled.structure_factor.size(); ++flat) {
    structure << flat;
    write_index_components(structure, momentum.modes[flat].indices);
    structure << ' ' << sampled.structure_factor[flat] << '\n';
  }
  for (std::size_t component = 0; component < linear_size; ++component) {
    const auto flat = axis_cut_flat_index(component, model);
    structure_cut << component << ' ' << momentum.modes[flat].wavevector.front() << ' '
                  << sampled.structure_factor[flat] << '\n';
  }
  auto onsite = output_file(directory / "onsite_occupation.dat");
  onsite << "# occupation probability\n";
  for (std::size_t occupation = 0; occupation < sampled.onsite_probability.size(); ++occupation) {
    onsite << occupation << ' ' << sampled.onsite_probability[occupation] << '\n';
  }
}

void write_imaginary_time(const std::filesystem::path &directory, const CommandLine &command_line,
                          const qmc::MomentumDistribution &momentum, const SampleAverages &sampled,
                          const std::optional<qmc::MatsubaraDensityCorrelations> &matsubara) {
  const auto volume = command_line.model.volume();
  auto output = output_file(directory / "imaginary_time_density.dat");
  output << "# tau_index tau flat displacement[0..d) connected_Cnn\n";
  auto onsite = output_file(directory / "imaginary_time_onsite.dat");
  onsite << "# tau_index tau Cnn(r=0,tau)\n";
  for (std::size_t time = 0; time < command_line.time_links; ++time) {
    const double tau = command_line.model.beta * static_cast<double>(time) /
                       static_cast<double>(command_line.time_links);
    for (std::size_t flat = 0; flat < volume; ++flat) {
      output << time << ' ' << tau << ' ' << flat;
      write_site_components(output, qmc::Site(momentum.modes[flat].indices.begin(),
                                              momentum.modes[flat].indices.end()));
      output << ' ' << sampled.connected_density[(time * volume) + flat] << '\n';
    }
    onsite << time << ' ' << tau << ' ' << sampled.connected_density[time * volume] << '\n';
  }

  auto geometry = output_file(directory / "retained_geometry.dat");
  geometry << "# tau_index tau mean_square_displacement return_probability\n";
  auto displacement = output_file(directory / "bridge_displacement.dat");
  displacement << "# tau_index tau flat displacement[0..d) probability\n";
  for (std::size_t time = 0; time < command_line.time_links; ++time) {
    const double tau = command_line.model.beta * static_cast<double>(time) /
                       static_cast<double>(command_line.time_links);
    geometry << time << ' ' << tau << ' ' << sampled.mean_square_displacement[time] << ' '
             << sampled.return_probability[time] << '\n';
    for (std::size_t flat = 0; flat < volume; ++flat) {
      displacement << time << ' ' << tau << ' ' << flat;
      write_site_components(displacement, qmc::Site(momentum.modes[flat].indices.begin(),
                                                    momentum.modes[flat].indices.end()));
      displacement << ' ' << sampled.displacement_probability[(time * volume) + flat] << '\n';
    }
  }

  auto transformed = output_file(directory / "matsubara_density.dat");
  transformed << "# n omega flat k[0..d) real imag\n";
  auto cut = output_file(directory / "matsubara_cut.dat");
  cut << "# n omega real imag; q=(2*pi/L,0,...) or q=0 when L=1\n";
  if (!matsubara.has_value()) {
    return;
  }
  const std::size_t cut_momentum = command_line.model.linear_size > 1 ? 1 : 0;
  for (std::size_t frequency = 0; frequency < matsubara->frequencies.size(); ++frequency) {
    for (std::size_t flat = 0; flat < volume; ++flat) {
      const auto value = matsubara->values[(frequency * volume) + flat];
      transformed << frequency << ' ' << matsubara->frequencies[frequency] << ' ' << flat;
      write_index_components(transformed, momentum.modes[flat].indices);
      transformed << ' ' << value.real() << ' ' << value.imag() << '\n';
    }
    const auto value = matsubara->values[(frequency * volume) + cut_momentum];
    cut << frequency << ' ' << matsubara->frequencies[frequency] << ' ' << value.real() << ' '
        << value.imag() << '\n';
  }
}

std::filesystem::path write_gnuplot_script(const std::filesystem::path &directory,
                                           const bool temperature_defined) {
  const auto script_path = directory / "ideal_observables.gnuplot";
  auto script = output_file(script_path);
  const auto file = [&directory](const std::string_view name) {
    return gnuplot_quote(directory / name);
  };
  script << "set datafile commentschars '#'\n"
            "set grid\n"
            "set key outside\n"
            "set terminal pngcairo size 1200,800 enhanced\n";

  script << "set output " << file("thermodynamics.png") << "\n";
  if (temperature_defined) {
    script << "set multiplot layout 2,2 title 'Exact canonical thermodynamics'\n"
           << "set xlabel 'N'; set ylabel 'energy'; plot " << file("thermodynamics.dat")
           << " using 1:2 with linespoints title 'F', '' using 1:3 with linespoints title 'E', "
              "'' using 1:6 with linespoints title 'mu'\n"
           << "set ylabel 'C'; plot " << file("thermodynamics.dat")
           << " using 1:4 with linespoints title 'heat capacity'\n"
           << "set ylabel 'S'; plot " << file("thermodynamics.dat")
           << " using 1:5 with linespoints title 'entropy'\n"
           << "unset multiplot\n";
  } else {
    script << "set title 'Canonical thermodynamics is undefined at beta=0'\n"
              "set xlabel 'N'; unset ylabel; plot 0 with lines title 'undefined'\n"
              "unset title\n";
  }

  script << "set output " << file("momentum_distribution.png") << "\n"
         << "set multiplot layout 2,1 title 'Momentum distribution and fluctuations'\n"
         << "set xlabel 'q_0'; set ylabel 'n(q)'; plot " << file("momentum_cut.dat")
         << " using 2:3 with linespoints title 'occupation'\n"
         << "set ylabel 'Var n(q)'; plot " << file("momentum_cut.dat")
         << " using 2:4 with linespoints title 'variance'\n"
         << "unset multiplot\n";

  script << "set output " << file("one_body_density_matrix.png") << "\n"
         << "set title 'One-body density matrix along axis 0'\n"
         << "set xlabel 'r_0'; set ylabel 'g_1(r)'; plot " << file("one_body_cut.dat")
         << " using 1:2 with linespoints title 'g_1'\n"
         << "unset title\n";

  script << "set output " << file("cycle_statistics.png") << "\n"
         << "unset title\n"
         << "set multiplot layout 2,2 title 'Permutation-cycle statistics'\n"
         << "set xlabel 'cycle length'; set ylabel '<m_l>'; plot " << file("cycle_statistics.dat")
         << " using 1:2 with linespoints title 'exact', '' using 1:5 with points title 'sampled'\n"
         << "set ylabel 'particle probability'; plot " << file("cycle_statistics.dat")
         << " using 1:4 with linespoints title 'exact P(l)'\n"
         << "set ylabel '<w_C^2 | l>'; plot " << file("cycle_statistics.dat")
         << " using 1:7 with linespoints title 'cycle winding'\n"
         << "set ylabel 'P(longest=l)'; plot " << file("cycle_statistics.dat")
         << " using 1:8 with boxes title 'longest cycle'\n"
         << "unset multiplot\n";

  script << "set output " << file("winding_statistics.png") << "\n"
         << "unset title\n"
         << "set multiplot layout 2,2 title 'Winding statistics and twist response'\n"
         << "set xlabel 'sample'; set ylabel 'W^2'; plot " << file("winding_samples.dat")
         << " using 1:2 with lines title 'W^2'\n"
         << "set xlabel 'W^2'; set ylabel 'probability'; set style fill solid 0.6; plot "
         << file("winding_squared_histogram.dat") << " using 1:2 with boxes title 'P(W^2)'\n"
         << "set xlabel 'axis'; set ylabel 'd^2 F / d phi^2'; ";
  if (temperature_defined) {
    script << "plot " << file("winding_moments.dat")
           << " using 1:4 with points title 'sampled', '' using 1:5 with points title 'exact'\n";
  } else {
    script << "plot 0 with lines title 'undefined at beta=0'\n";
  }
  script << "set xlabel 'axis'; set ylabel 'fourth cumulant'; plot " << file("winding_moments.dat")
         << " using 1:3 with points title 'kappa_4'\n"
         << "unset multiplot\n";

  script << "set output " << file("equal_time_density.png") << "\n"
         << "unset title\n"
         << "set multiplot layout 2,2 title 'Equal-time density observables'\n"
         << "set xlabel 'flat site'; set ylabel '<n_r>'; plot " << file("site_density.dat")
         << " using 1:2 with linespoints title 'density'\n"
         << "set xlabel 'r_0'; set ylabel 'g_2'; plot " << file("pair_correlation_cut.dat")
         << " using 1:2 with linespoints title 'g_2'\n"
         << "set xlabel 'q_0'; set ylabel 'S(q)'; plot " << file("structure_factor_cut.dat")
         << " using 2:3 with linespoints title 'structure factor'\n"
         << "set xlabel 'occupation'; set ylabel 'probability'; plot "
         << file("onsite_occupation.dat") << " using 1:2 with boxes title 'P(n)'\n"
         << "unset multiplot\n";

  script << "set output " << file("imaginary_time_correlations.png") << "\n"
         << "unset title\n"
         << "set multiplot layout 2,2 title 'Retained-grid correlations and geometry'\n"
         << "set xlabel 'tau'; set ylabel 'C_nn(0,tau)'; plot " << file("imaginary_time_onsite.dat")
         << " using 2:3 with linespoints title 'onsite connected correlation'\n"
         << "set xlabel 'omega_n'; set ylabel 'Re S(q,i omega_n)'; ";
  if (temperature_defined) {
    script << "plot " << file("matsubara_cut.dat")
           << " using 2:3 with impulses title 'retained-grid transform'\n";
  } else {
    script << "plot 0 with lines title 'undefined at beta=0'\n";
  }
  script << "set xlabel 'tau'; set ylabel 'mean-square displacement'; plot "
         << file("retained_geometry.dat") << " using 2:3 with linespoints title 'MSD'\n"
         << "set xlabel 'tau'; set ylabel 'return probability'; plot "
         << file("retained_geometry.dat")
         << " using 2:4 with linespoints title 'return probability'\n"
         << "unset multiplot\n";
  return script_path;
}

void print_summary(const CommandLine &command_line, const qmc::CanonicalEnsemble &canonical,
                   const std::optional<qmc::CanonicalThermodynamics> &thermodynamics,
                   const qmc::MomentumDistribution &momentum,
                   const std::vector<qmc::OneBodyDensityPoint> &density_matrix,
                   const qmc::ExactCycleStatistics &exact_cycles,
                   const std::vector<double> &twist_curvature, const SampleAverages &sampled,
                   const std::filesystem::path &directory) {
  const auto &model = command_line.model;
  const auto sample_count = static_cast<double>(command_line.samples);
  const double expected_density =
      static_cast<double>(model.particle_count) / static_cast<double>(model.volume());
  const double maximum_density_error = std::transform_reduce(
      sampled.site_density.begin(), sampled.site_density.end(), 0.0,
      [](const double left, const double right) { return std::max(left, right); },
      [expected_density](const double density) { return std::abs(density - expected_density); });
  const double expected_cycle_count = std::accumulate(exact_cycles.expected_cycle_count.begin(),
                                                      exact_cycles.expected_cycle_count.end(), 0.0);
  const double sampled_cycle_count =
      std::accumulate(sampled.cycle_count.begin(), sampled.cycle_count.end(), 0.0);
  const double mean_winding_squared = std::accumulate(sampled.winding_second_moment.begin(),
                                                      sampled.winding_second_moment.end(), 0.0);
  const double one_body_trace =
      density_matrix.empty() ? 0.0
                             : static_cast<double>(model.volume()) * density_matrix.front().value;
  const double one_body_sum =
      std::accumulate(density_matrix.begin(), density_matrix.end(), 0.0,
                      [](const double value, const qmc::OneBodyDensityPoint &point) {
                        return value + point.value;
                      });

  std::cout << std::setprecision(12);
  std::cout << "model: N=" << model.particle_count << " beta=" << model.beta
            << " L=" << model.linear_size << " d=" << model.dimension << " t=" << model.hopping
            << " M=" << command_line.time_links << " samples=" << command_line.samples << '\n';
  std::cout << "log Z_N = " << canonical.log_partition(model.particle_count) << '\n';
  if (thermodynamics.has_value()) {
    const auto particles = model.particle_count;
    std::cout << "thermodynamics: F=" << thermodynamics->free_energy[particles]
              << " E=" << thermodynamics->energy[particles]
              << " C=" << thermodynamics->heat_capacity[particles]
              << " S=" << thermodynamics->entropy[particles]
              << " mu_add=" << thermodynamics->addition_chemical_potential[particles] << '\n';
  } else {
    std::cout << "thermodynamics: undefined at beta=0 (canonical weights were still measured)\n";
  }
  std::cout << "momentum: sum_q n(q)="
            << std::accumulate(momentum.modes.begin(), momentum.modes.end(), 0.0,
                               [](const double value, const qmc::MomentumMode &mode) {
                                 return value + mode.occupation;
                               })
            << " N0=" << momentum.condensate_occupation << " N0/N=" << momentum.condensate_fraction
            << " coherence_length=" << momentum.coherence_length
            << " kinetic_energy=" << momentum.kinetic_energy << '\n';
  std::cout << "one-body: V*g1(0)=" << one_body_trace << " sum_r g1(r)=" << one_body_sum
            << " g1(max separation)=" << density_matrix[farthest_flat_index(model)].value
            << " condensate_density=" << momentum.condensate_density << '\n';
  std::cout << "cycles: exact <count>=" << expected_cycle_count
            << " sampled <count>=" << sampled_cycle_count
            << " macroscopic particle fraction=" << sampled.macroscopic_cycle_fraction << '\n';
  std::cout << "winding: <W^2>=" << mean_winding_squared
            << " P(W!=0)=" << sampled.nonzero_winding_probability << '\n';
  for (std::size_t axis = 0; axis < model.dimension; ++axis) {
    const double fourth_cumulant =
        sampled.winding_fourth_moment[axis] -
        (3.0 * sampled.winding_second_moment[axis] * sampled.winding_second_moment[axis]);
    std::cout << "  axis " << axis << ": <W_a^2>=" << sampled.winding_second_moment[axis]
              << " kappa4=" << fourth_cumulant;
    if (model.beta > 0.0) {
      std::cout << " sampled F''=" << sampled.winding_second_moment[axis] / model.beta
                << " exact F''=" << twist_curvature[axis];
    }
    std::cout << '\n';
  }
  std::cout << "density: N_check="
            << std::accumulate(sampled.site_density.begin(), sampled.site_density.end(), 0.0)
            << " max|<n_r>-N/V|=" << maximum_density_error
            << " <n^2>=" << sampled.mean_occupation_squared
            << " <n(n-1)>=" << sampled.mean_factorial_occupation << " dF/dU|0="
            << 0.5 * static_cast<double>(model.volume()) * sampled.mean_factorial_occupation
            << " S(0)=" << sampled.structure_factor.front() << '\n';
  std::cout << "retained-grid C_nn: points=" << command_line.time_links << " x " << model.volume()
            << " C_nn(0,0)=" << sampled.connected_density.front()
            << " MSD(beta-dtau)=" << sampled.mean_square_displacement.back()
            << " return(beta-dtau)=" << sampled.return_probability.back() << '\n';
  std::cout << "measurements written to " << directory << " (" << sample_count
            << " independent configurations)\n";
}

} // namespace

int main(const int argc, char **argv) {
  try {
    const auto command_line = parse_command_line(argc, argv);
    if (!command_line.has_value()) {
      return 0;
    }
    command_line->model.validate();
    if (command_line->time_links < 1) {
      throw std::invalid_argument("time_links must be positive");
    }

    const auto directory = std::filesystem::absolute(command_line->output_directory);
    std::filesystem::create_directories(directory);
    const qmc::CanonicalEnsemble canonical(command_line->model);
    std::optional<qmc::CanonicalThermodynamics> thermodynamics;
    std::vector<double> twist_curvature(command_line->model.dimension);
    if (command_line->model.beta > 0.0) {
      thermodynamics = qmc::canonical_thermodynamics(canonical);
      for (std::size_t axis = 0; axis < command_line->model.dimension; ++axis) {
        twist_curvature[axis] = qmc::twist_free_energy_curvature(canonical, axis);
      }
    }
    const auto momentum = qmc::momentum_distribution(canonical);
    const auto density_matrix = qmc::one_body_density_matrix(canonical);
    const auto exact_cycles = qmc::exact_cycle_statistics(canonical);

    qmc::Random random(command_line->seed);
    const auto sampled = sample_ensemble(*command_line, canonical, random);
    const qmc::ImaginaryTimeDensityCorrelations averaged_correlations(
        qmc::RetainedGrid(
            command_line->model.beta,
            qmc::TorusLayout(command_line->model.linear_size, command_line->model.dimension),
            command_line->time_links),
        sampled.connected_density);
    std::optional<qmc::MatsubaraDensityCorrelations> matsubara;
    if (command_line->model.beta > 0.0) {
      matsubara = qmc::retained_grid_matsubara_transform(averaged_correlations);
    }

    write_thermodynamics(directory, command_line->model, thermodynamics);
    write_momentum(directory, command_line->model, momentum);
    write_one_body(directory, command_line->model, density_matrix);
    write_cycles(directory, command_line->model, exact_cycles, sampled);
    write_winding(directory, *command_line, twist_curvature, sampled);
    write_density(directory, command_line->model, momentum, sampled);
    write_imaginary_time(directory, *command_line, momentum, sampled, matsubara);
    const auto script = write_gnuplot_script(directory, command_line->model.beta > 0.0);
    print_summary(*command_line, canonical, thermodynamics, momentum, density_matrix, exact_cycles,
                  twist_curvature, sampled, directory);

    if (command_line->plot) {
      const std::string command =
          shell_quote(command_line->gnuplot) + " " + shell_quote(script.string());
      const int status = std::system(command.c_str());
      if (status != 0) {
        throw std::runtime_error("gnuplot failed with status " + std::to_string(status));
      }
      std::cout << "gnuplot wrote PNG plots to " << directory << '\n';
    } else {
      std::cout << "run gnuplot " << script << " or pass --plot to render PNG plots\n";
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
