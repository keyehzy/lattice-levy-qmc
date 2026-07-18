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

if(BUILD_TESTING)
  add_test(NAME qmc_ideal_demo_help COMMAND qmc_ideal_demo --help)
  set_tests_properties(qmc_ideal_demo_help PROPERTIES LABELS unit)

  add_test(
    NAME qmc_ideal_demo_named_options
    COMMAND
      qmc_ideal_demo --particles 1 --beta 0 --time-links 1 --linear-size 1 --dimension 1
      --hopping 0 --seed 1
  )
  set_tests_properties(qmc_ideal_demo_named_options PROPERTIES LABELS unit)

  add_test(NAME qmc_ideal_demo_positional_options COMMAND qmc_ideal_demo 1 0 1 1 1 0 1)
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
endif()
