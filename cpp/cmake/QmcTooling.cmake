function(qmc_set_project_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(
      ${target}
      PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
    )
  endif()
endfunction()

function(qmc_enable_clang_tidy target)
  if(NOT QMC_ENABLE_CLANG_TIDY)
    return()
  endif()

  find_program(
    QMC_CLANG_TIDY_EXECUTABLE
    NAMES clang-tidy clang-tidy-21
    HINTS "/opt/homebrew/opt/llvm/bin"
  )
  if(NOT QMC_CLANG_TIDY_EXECUTABLE)
    message(FATAL_ERROR "QMC_ENABLE_CLANG_TIDY is ON, but clang-tidy was not found")
  endif()

  set_property(
    TARGET ${target}
    PROPERTY CXX_CLANG_TIDY
      "${QMC_CLANG_TIDY_EXECUTABLE};--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
  )
endfunction()

function(qmc_add_format_targets)
  find_program(QMC_CLANG_FORMAT_EXECUTABLE NAMES clang-format clang-format-21)
  if(NOT QMC_CLANG_FORMAT_EXECUTABLE)
    message(STATUS "clang-format not found; format targets will not be available")
    return()
  endif()

  file(
    GLOB_RECURSE QMC_FORMAT_FILES
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.hpp"
    "${PROJECT_SOURCE_DIR}/tests/*.cpp"
    "${PROJECT_SOURCE_DIR}/tests/*.h"
    "${PROJECT_SOURCE_DIR}/tests/*.hpp"
    "${PROJECT_SOURCE_DIR}/examples/*.cpp"
  )

  if(NOT QMC_FORMAT_FILES)
    message(STATUS "No C++ files found; format targets will be added with the first source file")
    return()
  endif()

  add_custom_target(
    format
    COMMAND "${QMC_CLANG_FORMAT_EXECUTABLE}" -i ${QMC_FORMAT_FILES}
    COMMENT "Formatting C++ sources"
    VERBATIM
  )
  add_custom_target(
    format-check
    COMMAND "${QMC_CLANG_FORMAT_EXECUTABLE}" --dry-run --Werror ${QMC_FORMAT_FILES}
    COMMENT "Checking C++ source formatting"
    VERBATIM
  )
endfunction()
