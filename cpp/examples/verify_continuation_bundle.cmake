if(NOT DEFINED PROGRAM OR NOT DEFINED OUTPUT_DIRECTORY)
  message(FATAL_ERROR "PROGRAM and OUTPUT_DIRECTORY are required")
endif()

file(REMOVE_RECURSE "${OUTPUT_DIRECTORY}")
file(MAKE_DIRECTORY "${OUTPUT_DIRECTORY}")
set(bundle "${OUTPUT_DIRECTORY}/density-continuation-v1")
set(trace "${OUTPUT_DIRECTORY}/scalar-trace.dat")
set(arguments
    --particles=1
    --beta=0.5
    --linear-size=2
    --dimension=1
    --hopping=0
    --interaction=0
    --seed=17
    --samples=4
    --burn-in=0
    --thin=1
    --segment-updates=0
    --cycle-updates=0
    --global-updates=0
    --stitch-updates=0
    --density-momenta=1
    --density-frequency-max=1
    --density-measurements-per-block=2
    "--density-continuation-dir=${bundle}"
    "--output=${trace}"
    --no-trace
)

execute_process(
  COMMAND "${PROGRAM}" ${arguments}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "continuation demo failed with ${result}\nstdout:\n${output}\nstderr:\n${error}")
endif()
if(NOT output MATCHES "density blocks = 2")
  message(FATAL_ERROR "demo did not report its completed block count\nstdout:\n${output}")
endif()
if(NOT output MATCHES "largest density standard error = 0")
  message(FATAL_ERROR "demo did not report its largest standard error\nstdout:\n${output}")
endif()
if(EXISTS "${trace}")
  message(FATAL_ERROR "--no-trace unexpectedly created ${trace}")
endif()

foreach(filename IN ITEMS manifest.tsv values.tsv covariance.tsv blocks.tsv)
  set(path "${bundle}/${filename}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "expected continuation output does not exist: ${path}")
  endif()
  file(SIZE "${path}" size)
  if(size EQUAL 0)
    message(FATAL_ERROR "expected continuation output is empty: ${path}")
  endif()
endforeach()

file(READ "${bundle}/manifest.tsv" manifest)
foreach(required IN ITEMS
        "schema_id\tdensity-continuation"
        "schema_version\t1"
        "basis\tbosonic_matsubara"
        "model_interaction\t0"
        "measurements_per_block\t2"
        "completed_block_count\t2"
        "sample_count\t4"
        "covariance_rank_status\trank_deficient_by_completed_block_count"
        "scalar_trace_retained\tfalse"
        "program\tqmc_interacting_demo")
  string(FIND "${manifest}" "${required}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "manifest is missing: ${required}")
  endif()
endforeach()

file(STRINGS "${bundle}/values.tsv" value_rows)
file(STRINGS "${bundle}/covariance.tsv" covariance_rows)
file(STRINGS "${bundle}/blocks.tsv" block_rows)
list(LENGTH value_rows value_row_count)
list(LENGTH covariance_rows covariance_row_count)
list(LENGTH block_rows block_row_count)
if(NOT value_row_count EQUAL 3)
  message(FATAL_ERROR "values.tsv has ${value_row_count} lines instead of 3")
endif()
if(NOT covariance_row_count EQUAL 5)
  message(FATAL_ERROR "covariance.tsv has ${covariance_row_count} lines instead of 5")
endif()
if(NOT block_row_count EQUAL 5)
  message(FATAL_ERROR "blocks.tsv has ${block_row_count} lines instead of 5")
endif()

file(SHA256 "${bundle}/manifest.tsv" manifest_before)
execute_process(
  COMMAND "${PROGRAM}" ${arguments}
  RESULT_VARIABLE existing_result
  OUTPUT_VARIABLE existing_output
  ERROR_VARIABLE existing_error
)
if(NOT existing_result EQUAL 1)
  message(
    FATAL_ERROR
    "existing destination returned ${existing_result} instead of 1\n"
    "stdout:\n${existing_output}\nstderr:\n${existing_error}"
  )
endif()
file(SHA256 "${bundle}/manifest.tsv" manifest_after)
if(NOT manifest_before STREQUAL manifest_after)
  message(FATAL_ERROR "existing continuation bundle was modified")
endif()

set(partial_bundle "${OUTPUT_DIRECTORY}/partial-bundle")
execute_process(
  COMMAND
    "${PROGRAM}" --particles=1 --beta=0.5 --linear-size=2 --dimension=1 --hopping=0
    --interaction=0 --samples=3 --burn-in=0 --segment-updates=0 --cycle-updates=0
    --global-updates=0 --stitch-updates=0 --density-momenta=1 --density-frequency-max=1
    --density-measurements-per-block=2 "--density-continuation-dir=${partial_bundle}" --no-trace
  RESULT_VARIABLE partial_result
  OUTPUT_VARIABLE partial_output
  ERROR_VARIABLE partial_error
)
if(NOT partial_result EQUAL 1)
  message(
    FATAL_ERROR
    "partial-block request returned ${partial_result} instead of 1\n"
    "stdout:\n${partial_output}\nstderr:\n${partial_error}"
  )
endif()
if(EXISTS "${partial_bundle}")
  message(FATAL_ERROR "invalid partial-block request created ${partial_bundle}")
endif()

set(zero_bundle "${OUTPUT_DIRECTORY}/zero-momentum-bundle")
execute_process(
  COMMAND
    "${PROGRAM}" --particles=1 --beta=0.5 --linear-size=2 --dimension=1 --hopping=0
    --interaction=0 --samples=4 --burn-in=0 --segment-updates=0 --cycle-updates=0
    --global-updates=0 --stitch-updates=0 --density-momenta=0 --density-frequency-max=1
    --density-measurements-per-block=2 "--density-continuation-dir=${zero_bundle}" --no-trace
  RESULT_VARIABLE zero_result
  OUTPUT_VARIABLE zero_output
  ERROR_VARIABLE zero_error
)
if(NOT zero_result EQUAL 1)
  message(
    FATAL_ERROR
    "q=0-only request returned ${zero_result} instead of 1\n"
    "stdout:\n${zero_output}\nstderr:\n${zero_error}"
  )
endif()
if(EXISTS "${zero_bundle}")
  message(FATAL_ERROR "q=0-only request created ${zero_bundle}")
endif()

set(retained_bundle "${OUTPUT_DIRECTORY}/retained-trace-bundle")
set(retained_trace "${OUTPUT_DIRECTORY}/retained-trace.dat")
set(retained_arguments ${arguments})
list(REMOVE_ITEM retained_arguments --no-trace)
list(REMOVE_ITEM retained_arguments "--density-continuation-dir=${bundle}")
list(REMOVE_ITEM retained_arguments "--output=${trace}")
list(
  APPEND
  retained_arguments
  "--density-continuation-dir=${retained_bundle}"
  "--output=${retained_trace}"
)
execute_process(
  COMMAND "${PROGRAM}" ${retained_arguments}
  RESULT_VARIABLE retained_result
  OUTPUT_VARIABLE retained_output
  ERROR_VARIABLE retained_error
)
if(NOT retained_result EQUAL 0)
  message(
    FATAL_ERROR
    "retained-trace continuation demo failed with ${retained_result}\n"
    "stdout:\n${retained_output}\nstderr:\n${retained_error}"
  )
endif()
if(NOT EXISTS "${retained_trace}")
  message(FATAL_ERROR "default continuation workflow did not retain ${retained_trace}")
endif()
file(SIZE "${retained_trace}" retained_trace_size)
if(retained_trace_size EQUAL 0)
  message(FATAL_ERROR "retained scalar trace is empty: ${retained_trace}")
endif()
file(READ "${retained_bundle}/manifest.tsv" retained_manifest)
string(FIND "${retained_manifest}" "scalar_trace_retained\ttrue" retained_flag)
if(retained_flag EQUAL -1)
  message(FATAL_ERROR "retained-trace manifest does not record scalar_trace_retained=true")
endif()
