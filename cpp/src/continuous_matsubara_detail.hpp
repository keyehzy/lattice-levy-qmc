#ifndef QMC_SRC_CONTINUOUS_MATSUBARA_DETAIL_HPP
#define QMC_SRC_CONTINUOUS_MATSUBARA_DETAIL_HPP

#include "qmc/matsubara_modes.hpp"

#include <complex>
#include <cstdint>

namespace qmc::detail {

// Exact event-time phase exp(i*2*pi*n*time/beta), with explicit seam values.
[[nodiscard]] std::complex<double> matsubara_time_phase(std::int64_t index, double time,
                                                        double beta);

// Exact integral of exp(i*2*pi*n*tau/beta) over [begin, end).
[[nodiscard]] std::complex<double> matsubara_interval_transform(std::int64_t index, double begin,
                                                                double end, double beta);

// Physical-site factor exp(-i*q*x), reducing every integer product modulo L
// before conversion to double.
[[nodiscard]] std::complex<double> matsubara_site_phase(const MatsubaraModeSet &modes,
                                                        std::size_t momentum, SiteId site);

} // namespace qmc::detail

#endif
