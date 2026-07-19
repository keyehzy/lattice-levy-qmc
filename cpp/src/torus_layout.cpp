#include "qmc/torus_layout.hpp"

#include "checked_math.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>

namespace qmc {

std::size_t SiteIdHash::operator()(const SiteId site) const noexcept {
  return std::hash<std::size_t>{}(site.value());
}

TorusLayout::TorusLayout(const Coord linear_size, const std::size_t dimension)
    : linear_size_(linear_size), volume_(checked_volume(linear_size, dimension)) {
  if (dimension > strides_.max_size()) {
    throw std::length_error("lattice dimension exceeds vector capacity");
  }

  const auto size = static_cast<std::size_t>(linear_size);
  std::size_t stride = 1;
  strides_.reserve(dimension);
  for (std::size_t axis = 0; axis < dimension; ++axis) {
    strides_.push_back(stride);
    stride *= size;
  }
}

std::size_t TorusLayout::checked_volume(const Coord linear_size, const std::size_t dimension) {
  if (linear_size < 1) {
    throw std::invalid_argument("linear_size must be positive");
  }
  if (dimension < 1) {
    throw std::invalid_argument("dimension must be positive");
  }
  const auto unsigned_size = static_cast<std::uint64_t>(linear_size);
  if (unsigned_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("linear_size exceeds size_t");
  }

  const auto size = static_cast<std::size_t>(linear_size);
  std::size_t volume = 1;
  for (std::size_t axis = 0; axis < dimension; ++axis) {
    volume = detail::checked_product(volume, size, "lattice volume exceeds size_t");
  }
  return volume;
}

std::size_t TorusLayout::reduced_component(const Coord coordinate) const noexcept {
  return static_cast<std::size_t>(reduce(coordinate));
}

Coord TorusLayout::reduce(const Coord coordinate) const noexcept {
  const Coord remainder = coordinate % linear_size_;
  return remainder < 0 ? remainder + linear_size_ : remainder;
}

std::vector<Coord> TorusLayout::reduce(const std::span<const Coord> site) const {
  std::vector<Coord> reduced(dimension());
  reduce_into(site, reduced);
  return reduced;
}

void TorusLayout::reduce_into(const std::span<const Coord> site,
                              const std::span<Coord> reduced) const {
  if (site.size() != dimension() || reduced.size() != dimension()) {
    throw std::invalid_argument("site dimension does not match the torus layout");
  }
  for (std::size_t axis = 0; axis < dimension(); ++axis) {
    reduced[axis] = reduce(site[axis]);
  }
}

SiteId TorusLayout::encode(const std::span<const Coord> site) const {
  if (site.size() != dimension()) {
    throw std::invalid_argument("site dimension does not match the torus layout");
  }
  std::size_t flat = 0;
  for (std::size_t axis = 0; axis < dimension(); ++axis) {
    if (site[axis] < 0 || site[axis] >= linear_size_) {
      throw std::invalid_argument("torus component is outside [0, L)");
    }
    flat += static_cast<std::size_t>(site[axis]) * strides_[axis];
  }
  return SiteId(flat);
}

SiteId TorusLayout::encode(const std::span<const std::size_t> components) const {
  if (components.size() != dimension()) {
    throw std::invalid_argument("component vector has the wrong dimension");
  }
  const auto size = static_cast<std::size_t>(linear_size_);
  std::size_t flat = 0;
  for (std::size_t axis = 0; axis < dimension(); ++axis) {
    if (components[axis] >= size) {
      throw std::invalid_argument("torus component is outside [0, L)");
    }
    flat += components[axis] * strides_[axis];
  }
  return SiteId(flat);
}

SiteId TorusLayout::encode_covering(const std::span<const Coord> site) const {
  if (site.size() != dimension()) {
    throw std::invalid_argument("site dimension does not match the torus layout");
  }
  std::size_t flat = 0;
  for (std::size_t axis = 0; axis < dimension(); ++axis) {
    flat += reduced_component(site[axis]) * strides_[axis];
  }
  return SiteId(flat);
}

void TorusLayout::validate_site_id(const SiteId site) const {
  if (site.value() >= volume_) {
    throw std::out_of_range("site identity is outside the torus layout");
  }
}

std::vector<std::size_t> TorusLayout::decode(const SiteId site) const {
  std::vector<std::size_t> components(dimension());
  decode_into(site, components);
  return components;
}

void TorusLayout::decode_into(const SiteId site, const std::span<std::size_t> components) const {
  validate_site_id(site);
  if (components.size() != dimension()) {
    throw std::invalid_argument("component output has the wrong dimension");
  }
  const auto size = static_cast<std::size_t>(linear_size_);
  std::size_t flat = site.value();
  for (std::size_t axis = 0; axis < dimension(); ++axis) {
    components[axis] = flat % size;
    flat /= size;
  }
}

SiteId TorusLayout::flat_displacement(const SiteId origin, const SiteId target) const {
  validate_site_id(origin);
  validate_site_id(target);
  const auto size = static_cast<std::size_t>(linear_size_);
  std::size_t origin_flat = origin.value();
  std::size_t target_flat = target.value();
  std::size_t displacement = 0;
  for (std::size_t axis = 0; axis < dimension(); ++axis) {
    const std::size_t origin_component = origin_flat % size;
    const std::size_t target_component = target_flat % size;
    const std::size_t component = target_component >= origin_component
                                      ? target_component - origin_component
                                      : size - (origin_component - target_component);
    displacement += component * strides_[axis];
    origin_flat /= size;
    target_flat /= size;
  }
  return SiteId(displacement);
}

SiteId TorusLayout::shifted(const SiteId site, const std::size_t axis,
                            const Coord displacement) const {
  validate_site_id(site);
  if (axis >= dimension()) {
    throw std::out_of_range("shift axis is outside the torus layout");
  }
  const auto size = static_cast<std::size_t>(linear_size_);
  const std::size_t old_component = (site.value() / strides_[axis]) % size;
  const std::size_t delta = reduced_component(displacement);
  const std::size_t distance_to_wrap = size - old_component;
  const std::size_t new_component =
      delta < distance_to_wrap ? old_component + delta : delta - distance_to_wrap;
  return SiteId(site.value() - (old_component * strides_[axis]) + (new_component * strides_[axis]));
}

bool TorusLayout::within_radius(const SiteId left, const SiteId right,
                                const std::size_t radius) const {
  validate_site_id(left);
  validate_site_id(right);
  const auto size = static_cast<std::size_t>(linear_size_);
  std::size_t left_flat = left.value();
  std::size_t right_flat = right.value();
  for (std::size_t axis = 0; axis < dimension(); ++axis) {
    const std::size_t left_component = left_flat % size;
    const std::size_t right_component = right_flat % size;
    const std::size_t direct = left_component >= right_component ? left_component - right_component
                                                                 : right_component - left_component;
    if (std::min(direct, size - direct) > radius) {
      return false;
    }
    left_flat /= size;
    right_flat /= size;
  }
  return true;
}

std::vector<SiteId> TorusLayout::neighbors_within_radius(const SiteId center,
                                                         const std::size_t radius) const {
  validate_site_id(center);
  const auto size = static_cast<std::size_t>(linear_size_);
  const std::size_t axis_width = radius >= size / 2 ? size : (2 * radius) + 1;
  std::size_t count = 1;
  for (std::size_t axis = 0; axis < dimension(); ++axis) {
    count = detail::checked_product(count, axis_width, "torus neighborhood size exceeds size_t");
  }

  std::vector<SiteId> neighbors;
  if (count > neighbors.max_size()) {
    throw std::length_error("torus neighborhood exceeds vector capacity");
  }
  neighbors.reserve(count);
  const std::vector<std::size_t> center_components = decode(center);
  const std::size_t negative_width = axis_width == size ? 0 : radius;
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
    std::size_t divisor = count;
    std::size_t flat = 0;
    for (std::size_t axis = 0; axis < dimension(); ++axis) {
      divisor /= axis_width;
      const std::size_t digit = (ordinal / divisor) % axis_width;
      std::size_t component = digit;
      if (axis_width != size) {
        if (digit < negative_width) {
          const std::size_t magnitude = negative_width - digit;
          component = magnitude <= center_components[axis]
                          ? center_components[axis] - magnitude
                          : size - (magnitude - center_components[axis]);
        } else {
          const std::size_t positive = digit - negative_width;
          const std::size_t distance_to_wrap = size - center_components[axis];
          component = positive < distance_to_wrap ? center_components[axis] + positive
                                                  : positive - distance_to_wrap;
        }
      }
      flat += component * strides_[axis];
    }
    neighbors.emplace_back(flat);
  }
  return neighbors;
}

} // namespace qmc
