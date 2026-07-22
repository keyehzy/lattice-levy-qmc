#ifndef QMC_TESTS_CONTINUOUS_TEST_FIXTURES_HPP
#define QMC_TESTS_CONTINUOUS_TEST_FIXTURES_HPP

#include "qmc/continuous_configuration.hpp"

namespace qmc::test {

inline ContinuousConfiguration coincident_seam_configuration() {
  const Model model(ModelParameters{
      .particle_count = 2,
      .beta = 1.0,
      .linear_size = 5,
      .dimension = 1,
      .hopping = 1.0,
  });
  const ContinuousPath first(1.0, {0}, {1},
                             {{.time = 0.0, .axis = 0, .direction = 1},
                              {.time = 0.25, .axis = 0, .direction = 1},
                              {.time = 0.25, .axis = 0, .direction = -1},
                              {.time = 0.75, .axis = 0, .direction = 1},
                              {.time = 1.0, .axis = 0, .direction = -1}});
  const ContinuousPath second(1.0, {1}, {0},
                              {{.time = 0.0, .axis = 0, .direction = -1},
                               {.time = 0.25, .axis = 0, .direction = 1},
                               {.time = 0.5, .axis = 0, .direction = 1},
                               {.time = 0.75, .axis = 0, .direction = -1},
                               {.time = 0.75, .axis = 0, .direction = 1},
                               {.time = 1.0, .axis = 0, .direction = -1},
                               {.time = 1.0, .axis = 0, .direction = -1}});
  return ContinuousConfiguration(model, Permutation({1, 0}), {first, second});
}

} // namespace qmc::test

#endif
