cmake_minimum_required(VERSION 3.24)

project(
  qmc
  VERSION 0.5.0
  DESCRIPTION "Canonical continuous-time lattice Levy QMC"
  LANGUAGES CXX
)

configure_file(
  include/qmc/version.hpp.in
  generated/include/qmc/version.hpp
  @ONLY
)

include(CTest)
find_package(GSL 2.7 REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(QmcTooling)

option(QMC_ENABLE_CLANG_TIDY "Run clang-tidy while compiling project targets" OFF)
option(QMC_ENABLE_SANITIZERS "Build project targets with address and undefined sanitizers" OFF)
option(QMC_BUILD_EXAMPLES "Build example programs" ON)
option(QMC_BUILD_STATISTICAL_TESTS "Build slower distribution-level tests" OFF)

add_library(
  qmc_ideal
  src/canonical_recursion.cpp
  src/configuration.cpp
  src/free_boson.cpp
  src/free_numerics.cpp
  src/model.cpp
  src/observables.cpp
  src/permutation.cpp
  src/random.cpp
  src/torus_layout.cpp
)
add_library(qmc::ideal ALIAS qmc_ideal)

target_compile_features(qmc_ideal PUBLIC cxx_std_20)
target_include_directories(
  qmc_ideal
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>"
    "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(qmc_ideal PRIVATE GSL::gsl)
qmc_set_project_warnings(qmc_ideal)
qmc_enable_clang_tidy(qmc_ideal)
qmc_enable_sanitizers(qmc_ideal)

add_library(
  qmc_interacting
  src/accepted_chain_state.cpp
  src/continuous_configuration.cpp
  src/interacting_model.cpp
  src/interacting_sampler.cpp
  src/interaction.cpp
  src/occupancy_index.cpp
  src/path.cpp
  src/path_cursor.cpp
  src/stitch_matching.cpp
  src/stitch_seam_context.cpp
  src/stitch_selection.cpp
)
add_library(qmc::interacting ALIAS qmc_interacting)

target_compile_features(qmc_interacting PUBLIC cxx_std_20)
target_include_directories(
  qmc_interacting
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>"
    "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(qmc_interacting PUBLIC qmc::ideal)
qmc_set_project_warnings(qmc_interacting)
qmc_enable_clang_tidy(qmc_interacting)
qmc_enable_sanitizers(qmc_interacting)

if(QMC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

qmc_add_format_targets()
