#ifndef QMC_EXAMPLES_CONTINUATION_BUNDLE_HPP
#define QMC_EXAMPLES_CONTINUATION_BUNDLE_HPP

#include "qmc/continuous_observables.hpp"
#include "qmc/interacting_model.hpp"
#include "qmc/interacting_sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace qmc::example {

// Complete provenance for the explicit interacting-demo advance schedule.
// One measurement advance applies random_seam_stitch followed by sweep.
struct DensityContinuationRunProvenance {
  InteractingModel model;
  std::uint64_t seed;
  std::size_t burn_in_sweeps;
  std::size_t thinning_sweeps;
  SweepOptions sweep;
  RandomSeamStitchOptions random_seam_stitch;
  bool scalar_trace_retained;
  std::string program;
  std::string program_version;
};

// Read-only destination validation used before sampler construction. The
// destination must not exist and its parent must already be a directory.
void validate_density_continuation_bundle_destination(const std::filesystem::path &destination);

// Writes density-continuation-v1 as four UTF-8 TSV files. All series and
// provenance data are validated before a sibling temporary directory is
// created. Existing destinations are rejected; successful publication renames
// the complete sibling directory into place.
void write_density_continuation_bundle(const std::filesystem::path &destination,
                                       const DensityMatsubaraBlockSeries &series,
                                       const DensityContinuationRunProvenance &provenance);

} // namespace qmc::example

#endif
