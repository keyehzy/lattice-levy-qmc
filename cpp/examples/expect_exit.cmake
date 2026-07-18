if(NOT DEFINED PROGRAM OR NOT DEFINED EXPECTED_EXIT_CODE OR NOT DEFINED ARGUMENT)
  message(FATAL_ERROR "PROGRAM, EXPECTED_EXIT_CODE, and ARGUMENT are required")
endif()

execute_process(
  COMMAND "${PROGRAM}" "${ARGUMENT}"
  RESULT_VARIABLE actual_exit_code
  OUTPUT_VARIABLE program_stdout
  ERROR_VARIABLE program_stderr
)

if(NOT actual_exit_code EQUAL EXPECTED_EXIT_CODE)
  message(
    FATAL_ERROR
    "Expected exit code ${EXPECTED_EXIT_CODE}, got ${actual_exit_code}\n"
    "stdout:\n${program_stdout}\n"
    "stderr:\n${program_stderr}"
  )
endif()
