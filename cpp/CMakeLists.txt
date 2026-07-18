cmake_minimum_required(VERSION 3.24)

project(
  qmc_ideal
  VERSION 0.1.0
  DESCRIPTION "Ideal-gas lattice Levy QMC"
  LANGUAGES CXX
)

include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(QmcTooling)

option(QMC_ENABLE_CLANG_TIDY "Run clang-tidy while compiling project targets" OFF)

add_library(qmc_ideal INTERFACE)
add_library(qmc::ideal ALIAS qmc_ideal)

target_compile_features(qmc_ideal INTERFACE cxx_std_20)
target_include_directories(
  qmc_ideal
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

qmc_add_format_targets()
