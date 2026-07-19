#ifndef QMC_TORUS_LAYOUT_HPP
#define QMC_TORUS_LAYOUT_HPP

#include "qmc/model.hpp"

#include <compare>
#include <cstddef>
#include <span>
#include <vector>

namespace qmc {

// Strong flat identity for one physical site. A SiteId is meaningful only with
// the TorusLayout that produced it.
class SiteId {
public:
  explicit constexpr SiteId(const std::size_t value) noexcept : value_(value) {}

  [[nodiscard]] constexpr std::size_t value() const noexcept { return value_; }

  auto operator<=>(const SiteId &) const = default;

private:
  std::size_t value_;
};

struct SiteIdHash {
  [[nodiscard]] std::size_t operator()(SiteId site) const noexcept;
};

// Checked row-major geometry for an L^d periodic lattice. Axis zero is the
// least-significant base-L digit in every flat site identity.
class TorusLayout {
public:
  TorusLayout(Coord linear_size, std::size_t dimension);

  // Computes L^d without allocating layout storage.
  [[nodiscard]] static std::size_t checked_volume(Coord linear_size, std::size_t dimension);

  [[nodiscard]] Coord linear_size() const noexcept { return linear_size_; }
  [[nodiscard]] std::size_t dimension() const noexcept { return strides_.size(); }
  [[nodiscard]] std::size_t volume() const noexcept { return volume_; }
  [[nodiscard]] std::span<const std::size_t> strides() const noexcept { return strides_; }

  [[nodiscard]] Coord reduce(Coord coordinate) const noexcept;
  [[nodiscard]] std::vector<Coord> reduce(std::span<const Coord> site) const;
  void reduce_into(std::span<const Coord> site, std::span<Coord> reduced) const;

  // Encodes an already reduced site and rejects components outside [0, L).
  [[nodiscard]] SiteId encode(std::span<const Coord> site) const;
  [[nodiscard]] SiteId encode(std::span<const std::size_t> components) const;
  // Reduces arbitrary covering-space coordinates before encoding.
  [[nodiscard]] SiteId encode_covering(std::span<const Coord> site) const;

  [[nodiscard]] std::vector<std::size_t> decode(SiteId site) const;
  void decode_into(SiteId site, std::span<std::size_t> components) const;

  // Flat physical displacement target-origin, reduced independently per axis.
  [[nodiscard]] SiteId flat_displacement(SiteId origin, SiteId target) const;
  // Applies one covering-space coordinate displacement and reduces on the torus.
  [[nodiscard]] SiteId shifted(SiteId site, std::size_t axis, Coord displacement) const;

  [[nodiscard]] bool within_radius(SiteId left, SiteId right, std::size_t radius) const;
  // Returns every distinct site in the periodic Chebyshev-radius neighborhood.
  // Ordering follows nested axes with axis zero outermost.
  [[nodiscard]] std::vector<SiteId> neighbors_within_radius(SiteId center,
                                                            std::size_t radius) const;

  bool operator==(const TorusLayout &) const = default;

private:
  void validate_site_id(SiteId site) const;
  [[nodiscard]] std::size_t reduced_component(Coord coordinate) const noexcept;

  Coord linear_size_;
  std::size_t volume_;
  std::vector<std::size_t> strides_;
};

} // namespace qmc

#endif
