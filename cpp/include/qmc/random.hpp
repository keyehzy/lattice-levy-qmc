#ifndef QMC_RANDOM_HPP
#define QMC_RANDOM_HPP

#include "qmc/model.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace qmc {

class Random {
public:
  explicit Random(std::uint64_t seed);

  // Uniform draws on [0, 1) and (0, 1), respectively.
  [[nodiscard]] double uniform_unit();
  [[nodiscard]] double uniform_open();
  // Uniform integer draw on [0, upper_exclusive).
  [[nodiscard]] std::uint64_t uniform_index(std::uint64_t upper_exclusive);
  [[nodiscard]] std::uint64_t binomial(std::uint64_t trials, double probability);
  // Draws an index proportional to finite nonnegative, not necessarily normalized weights.
  [[nodiscard]] std::size_t discrete_index(std::span<const double> weights);
  void shuffle(std::span<ParticleId> labels);

private:
  std::mt19937_64 engine_;
};

} // namespace qmc

#endif
