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

add_executable(qmc_scaffold_test scaffold_test.cpp)
target_link_libraries(qmc_scaffold_test PRIVATE qmc::ideal GTest::gtest_main)
qmc_set_project_warnings(qmc_scaffold_test)
qmc_enable_clang_tidy(qmc_scaffold_test)

gtest_discover_tests(qmc_scaffold_test)
