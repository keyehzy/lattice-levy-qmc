#ifndef QMC_PERMUTATION_HPP
#define QMC_PERMUTATION_HPP

#include "qmc/model.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qmc {

using Cycle = std::vector<ParticleId>;

// Validated particle-label permutation. Construction rejects non-bijections
// with invalid_argument. Successors are authoritative; cycles are rooted and
// ordered by increasing label in a read-only derived cache.
class Permutation {
public:
  Permutation() = default;
  explicit Permutation(std::vector<ParticleId> successors);

  [[nodiscard]] bool empty() const noexcept { return successors_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return successors_.size(); }
  // Throws out_of_range when label is not part of the permutation.
  [[nodiscard]] ParticleId successor(ParticleId label) const;
  [[nodiscard]] std::span<const ParticleId> successors() const noexcept { return successors_; }
  [[nodiscard]] std::span<const Cycle> cycles() const noexcept { return cycles_; }

  [[nodiscard]] bool operator==(const Permutation &other) const noexcept {
    return successors_ == other.successors_;
  }

private:
  std::vector<ParticleId> successors_;
  std::vector<Cycle> cycles_;
};

} // namespace qmc

#endif
