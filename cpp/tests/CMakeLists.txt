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
  test_configuration.cpp
  test_free_boson.cpp
  test_free_numerics.cpp
  test_model.cpp
)
target_link_libraries(qmc_unit_tests PRIVATE qmc::ideal GTest::gtest_main)
qmc_set_project_warnings(qmc_unit_tests)
qmc_enable_clang_tidy(qmc_unit_tests)
qmc_enable_sanitizers(qmc_unit_tests)
gtest_discover_tests(qmc_unit_tests PROPERTIES LABELS unit)

if(QMC_BUILD_STATISTICAL_TESTS)
  add_executable(qmc_statistical_tests test_sampling_statistical.cpp)
  target_link_libraries(qmc_statistical_tests PRIVATE qmc::ideal GTest::gtest_main)
  qmc_set_project_warnings(qmc_statistical_tests)
  qmc_enable_clang_tidy(qmc_statistical_tests)
  qmc_enable_sanitizers(qmc_statistical_tests)
  gtest_discover_tests(qmc_statistical_tests PROPERTIES LABELS statistical)
endif()
