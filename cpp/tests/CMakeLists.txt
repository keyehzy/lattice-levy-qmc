include(FetchContent)

find_package(GTest 1.17 CONFIG QUIET)

if(NOT GTest_FOUND)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 52eb8108c5bdec04579160ae17225d66034bd723
  )

  if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

add_executable(
  qmc_unit_tests
  test_adaptive_discrete_support.cpp
  test_canonical_recursion.cpp
  test_configuration.cpp
  test_free_boson.cpp
  test_free_numerics.cpp
  test_matsubara_modes.cpp
  test_model.cpp
  test_observables.cpp
  test_random.cpp
  test_torus_layout.cpp
  test_version.cpp
)
target_link_libraries(qmc_unit_tests PRIVATE qmc::ideal GTest::gtest_main)
target_include_directories(qmc_unit_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
qmc_set_project_warnings(qmc_unit_tests)
qmc_enable_sanitizers(qmc_unit_tests)
gtest_discover_tests(qmc_unit_tests PROPERTIES LABELS unit)

add_executable(
  qmc_interacting_unit_tests
  test_continuous_configuration.cpp
  test_continuous_observables.cpp
  test_interacting_sampler.cpp
  test_interaction.cpp
  test_occupancy_index.cpp
  test_path.cpp
  test_stitch_proposal.cpp
)
target_link_libraries(qmc_interacting_unit_tests PRIVATE qmc::interacting GTest::gtest_main)
target_include_directories(qmc_interacting_unit_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
qmc_set_project_warnings(qmc_interacting_unit_tests)
qmc_enable_sanitizers(qmc_interacting_unit_tests)
gtest_discover_tests(qmc_interacting_unit_tests PROPERTIES LABELS unit)

if(TARGET qmc_continuation_bundle)
  target_sources(
    qmc_interacting_unit_tests
    PRIVATE
      test_continuation_bundle.cpp
      "${PROJECT_SOURCE_DIR}/examples/continuation_bundle.cpp"
  )
  target_include_directories(qmc_interacting_unit_tests PRIVATE "${PROJECT_SOURCE_DIR}/examples")
endif()

if(QMC_BUILD_STATISTICAL_TESTS)
  add_executable(
    qmc_statistical_tests
    test_interacting_statistical.cpp
    test_sampling_statistical.cpp
    "${PROJECT_SOURCE_DIR}/examples/continuation_bundle.cpp"
  )
  target_link_libraries(qmc_statistical_tests PRIVATE qmc::interacting GTest::gtest_main)
  target_include_directories(qmc_statistical_tests PRIVATE "${PROJECT_SOURCE_DIR}/examples")
  qmc_set_project_warnings(qmc_statistical_tests)
  qmc_enable_sanitizers(qmc_statistical_tests)
  gtest_discover_tests(qmc_statistical_tests PROPERTIES LABELS statistical)
endif()
