include(FetchContent)

find_package(cxxopts 3.3 CONFIG QUIET)

if(NOT cxxopts_FOUND)
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG 44380e5a44706ab7347f400698c703eb2a196202
  )
  FetchContent_MakeAvailable(cxxopts)
endif()

add_executable(qmc_ideal_demo ideal_demo.cpp)
target_link_libraries(qmc_ideal_demo PRIVATE qmc::ideal cxxopts::cxxopts)
qmc_set_project_warnings(qmc_ideal_demo)
qmc_enable_clang_tidy(qmc_ideal_demo)
qmc_enable_sanitizers(qmc_ideal_demo)

add_executable(qmc_interacting_demo interacting_demo.cpp)
target_link_libraries(qmc_interacting_demo PRIVATE qmc::interacting cxxopts::cxxopts)
qmc_set_project_warnings(qmc_interacting_demo)
qmc_enable_clang_tidy(qmc_interacting_demo)
qmc_enable_sanitizers(qmc_interacting_demo)

if(BUILD_TESTING)
  set(QMC_DEMO_TEST_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test-observables")
  add_test(NAME qmc_ideal_demo_help COMMAND qmc_ideal_demo --help)
  set_tests_properties(qmc_ideal_demo_help PROPERTIES LABELS unit)

  add_test(
    NAME qmc_ideal_demo_named_options
    COMMAND
      qmc_ideal_demo --particles 1 --beta 0 --time-links 1 --linear-size 1 --dimension 1
      --hopping 0 --seed 1 --samples 1 --output-dir "${QMC_DEMO_TEST_OUTPUT}/named"
  )
  set_tests_properties(qmc_ideal_demo_named_options PROPERTIES LABELS unit)

  add_test(
    NAME qmc_ideal_demo_positional_options
    COMMAND
      qmc_ideal_demo 1 0 1 1 1 0 1 --samples 1 --output-dir
      "${QMC_DEMO_TEST_OUTPUT}/positional"
  )
  set_tests_properties(qmc_ideal_demo_positional_options PROPERTIES LABELS unit)

  add_test(
    NAME qmc_ideal_demo_cli_error
    COMMAND
      "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:qmc_ideal_demo>" -DEXPECTED_EXIT_CODE=2
      -DARGUMENT=--not-an-option -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake"
  )
  set_tests_properties(qmc_ideal_demo_cli_error PROPERTIES LABELS unit)

  add_test(
    NAME qmc_ideal_demo_model_error
    COMMAND
      "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:qmc_ideal_demo>" -DEXPECTED_EXIT_CODE=1
      -DARGUMENT=--time-links=0 -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_exit.cmake"
  )
  set_tests_properties(qmc_ideal_demo_model_error PROPERTIES LABELS unit)

  find_program(QMC_GNUPLOT_EXECUTABLE NAMES gnuplot)
  add_test(
    NAME qmc_ideal_demo_observable_outputs
    COMMAND
      "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:qmc_ideal_demo>"
      "-DOUTPUT_DIRECTORY=${QMC_DEMO_TEST_OUTPUT}/complete"
      "-DGNUPLOT=${QMC_GNUPLOT_EXECUTABLE}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/verify_observable_outputs.cmake"
  )
  set_tests_properties(qmc_ideal_demo_observable_outputs PROPERTIES LABELS unit)

  add_test(NAME qmc_interacting_demo_help COMMAND qmc_interacting_demo --help)
  set_tests_properties(qmc_interacting_demo_help PROPERTIES LABELS unit)

  add_test(
    NAME qmc_interacting_demo_smoke
    COMMAND
      qmc_interacting_demo --particles=2 --beta=0.8 --linear-size=3 --dimension=1
      --hopping=1 --interaction=1.2 --samples=4 --burn-in=2 --segment-updates=2
      --cycle-updates=1 --global-updates=1 --stitch-updates=0
      --output=${CMAKE_CURRENT_BINARY_DIR}/interacting-demo-smoke.dat
  )
  set_tests_properties(qmc_interacting_demo_smoke PROPERTIES LABELS unit)
endif()
