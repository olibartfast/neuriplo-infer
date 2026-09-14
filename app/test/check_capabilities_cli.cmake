# Asserts the capabilities version the binary itself publishes.
#
# Matching a regular expression against the raw output cannot do this: the
# document carries a second, unrelated "schema_version" inside
# diagnostics.run_report, so a pattern that looks like a version check can pass
# on the nested one while the top-level version is anything at all. CMake's own
# JSON reader answers the actual question and needs no Python.
#
# Run as: cmake -DNEURIPLO_INFER_BIN=<path> -DEXPECTED_VERSION=<n> -P this.cmake

if(NOT DEFINED NEURIPLO_INFER_BIN OR NOT DEFINED EXPECTED_VERSION)
    message(FATAL_ERROR "NEURIPLO_INFER_BIN and EXPECTED_VERSION are required")
endif()

execute_process(
    COMMAND "${NEURIPLO_INFER_BIN}" --capabilities
    OUTPUT_VARIABLE capabilities
    ERROR_VARIABLE diagnostics
    RESULT_VARIABLE exit_code
)

if(NOT exit_code EQUAL 0)
    message(FATAL_ERROR "--capabilities exited with ${exit_code}\n${diagnostics}")
endif()

string(JSON version ERROR_VARIABLE json_error GET "${capabilities}" schema_version)
if(json_error)
    message(FATAL_ERROR "--capabilities did not publish a top-level schema_version: ${json_error}")
endif()

if(NOT version EQUAL EXPECTED_VERSION)
    message(FATAL_ERROR
        "--capabilities publishes schema_version ${version}, expected ${EXPECTED_VERSION}.\n"
        "Bump docs/capabilities.schema.json and the consumers with it, or fix the emitter.")
endif()

# The nested run-report version is a separate contract; assert it is there so
# the two cannot silently collapse into one.
string(JSON run_report_version ERROR_VARIABLE nested_error
    GET "${capabilities}" diagnostics run_report schema_version)
if(nested_error)
    message(FATAL_ERROR "diagnostics.run_report.schema_version is missing: ${nested_error}")
endif()

message(STATUS
    "capabilities schema_version ${version}, run_report schema_version ${run_report_version}")
