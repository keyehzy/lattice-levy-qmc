if(NOT DEFINED PROGRAM OR NOT DEFINED OUTPUT_DIRECTORY)
  message(FATAL_ERROR "PROGRAM and OUTPUT_DIRECTORY are required")
endif()

file(REMOVE_RECURSE "${OUTPUT_DIRECTORY}")
set(arguments
    --particles 3
    --beta 0.8
    --time-links 4
    --linear-size 3
    --dimension 2
    --hopping 0.7
    --seed 41
    --samples 8
    --output-dir "${OUTPUT_DIRECTORY}"
)
if(DEFINED GNUPLOT AND NOT GNUPLOT STREQUAL "" AND NOT GNUPLOT MATCHES "-NOTFOUND$")
  list(APPEND arguments --plot --gnuplot "${GNUPLOT}")
endif()

execute_process(
  COMMAND "${PROGRAM}" ${arguments}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "demo failed with ${result}\nstdout:\n${output}\nstderr:\n${error}")
endif()

set(required_files
    thermodynamics.dat
    momentum_distribution.dat
    one_body_density_matrix.dat
    cycle_statistics.dat
    cycle_geometry_samples.dat
    winding_conditioned_cycles.dat
    winding_histogram.dat
    winding_moments.dat
    site_density.dat
    pair_correlation.dat
    static_structure_factor.dat
    onsite_occupation.dat
    imaginary_time_density.dat
    retained_geometry.dat
    bridge_displacement.dat
    matsubara_density.dat
    ideal_observables.gnuplot
)
if(DEFINED GNUPLOT AND NOT GNUPLOT STREQUAL "" AND NOT GNUPLOT MATCHES "-NOTFOUND$")
  list(APPEND required_files
       thermodynamics.png
       momentum_distribution.png
       one_body_density_matrix.png
       cycle_statistics.png
       winding_statistics.png
       equal_time_density.png
       imaginary_time_correlations.png
  )
endif()

foreach(filename IN LISTS required_files)
  set(path "${OUTPUT_DIRECTORY}/${filename}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "expected output does not exist: ${path}")
  endif()
  file(SIZE "${path}" size)
  if(size EQUAL 0)
    message(FATAL_ERROR "expected output is empty: ${path}")
  endif()
endforeach()
