#include "qmc/permutation.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qmc {

Permutation::Permutation(std::vector<ParticleId> successors) : successors_(std::move(successors)) {
  if (successors_.size() > std::numeric_limits<ParticleId>::max()) {
    throw std::invalid_argument("permutation size exceeds the ParticleId range");
  }
  std::vector<bool> appeared(successors_.size(), false);
  for (const ParticleId successor_label : successors_) {
    const auto index = static_cast<std::size_t>(successor_label);
    if (index >= successors_.size() || appeared[index]) {
      throw std::invalid_argument("permutation successors must form a bijection");
    }
    appeared[index] = true;
  }

  std::vector<bool> seen(successors_.size(), false);
  for (std::size_t root = 0; root < successors_.size(); ++root) {
    if (seen[root]) {
      continue;
    }
    Cycle cycle;
    std::size_t current = root;
    while (!seen[current]) {
      seen[current] = true;
      cycle.push_back(static_cast<ParticleId>(current));
      current = static_cast<std::size_t>(successors_[current]);
    }
    if (current != root) {
      throw std::logic_error("validated permutation did not close at its cycle root");
    }
    cycles_.push_back(std::move(cycle));
  }
}

ParticleId Permutation::successor(const ParticleId label) const {
  const auto index = static_cast<std::size_t>(label);
  if (index >= successors_.size()) {
    throw std::out_of_range("permutation label is out of range");
  }
  return successors_[index];
}

} // namespace qmc
