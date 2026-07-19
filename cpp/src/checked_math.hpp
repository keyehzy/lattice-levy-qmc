#ifndef QMC_SRC_CHECKED_MATH_HPP
#define QMC_SRC_CHECKED_MATH_HPP

#include "qmc/model.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qmc::detail {

inline std::uint64_t coordinate_offset(const Coord value) {
  return static_cast<std::uint64_t>(value) -
         static_cast<std::uint64_t>(std::numeric_limits<Coord>::min());
}

inline Coord coordinate_from_offset(const std::uint64_t offset) {
  constexpr auto sign_offset = std::uint64_t{1} << 63U;
  if (offset < sign_offset) {
    return std::numeric_limits<Coord>::min() + static_cast<Coord>(offset);
  }
  return static_cast<Coord>(offset - sign_offset);
}

// The boolean is true when b-a is nonnegative; the integer is |b-a|.
inline std::pair<bool, std::uint64_t> displacement(const Coord a, const Coord b) {
  const auto a_offset = coordinate_offset(a);
  const auto b_offset = coordinate_offset(b);
  if (b_offset >= a_offset) {
    return {true, b_offset - a_offset};
  }
  return {false, a_offset - b_offset};
}

inline Coord checked_add(const Coord left, const Coord right, const char *description) {
  if (right > 0 && left > std::numeric_limits<Coord>::max() - right) {
    throw std::overflow_error(description);
  }
  if (right < 0 && left < std::numeric_limits<Coord>::min() - right) {
    throw std::overflow_error(description);
  }
  return left + right;
}

inline Coord checked_subtract(const Coord left, const Coord right, const char *description) {
  if (right > 0 && left < std::numeric_limits<Coord>::min() + right) {
    throw std::overflow_error(description);
  }
  if (right < 0 && left > std::numeric_limits<Coord>::max() + right) {
    throw std::overflow_error(description);
  }
  return left - right;
}

inline Coord checked_scale(const Coord scale, const Coord value, const char *description) {
  if (scale < 0) {
    throw std::logic_error("coordinate scale must be nonnegative");
  }
  if (scale == 0 || value == 0) {
    return 0;
  }
  if (value > 0 && scale > std::numeric_limits<Coord>::max() / value) {
    throw std::overflow_error(description);
  }
  if (value < 0 && value < std::numeric_limits<Coord>::min() / scale) {
    throw std::overflow_error(description);
  }
  return scale * value;
}

inline std::size_t checked_add_size(const std::size_t left, const std::size_t right,
                                    const char *description) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(description);
  }
  return left + right;
}

inline std::size_t checked_product(const std::size_t left, const std::size_t right,
                                   const char *description) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(description);
  }
  return left * right;
}

inline std::size_t checked_size(const std::uint64_t value, const char *description) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(description);
  }
  return static_cast<std::size_t>(value);
}

} // namespace qmc::detail

#endif
