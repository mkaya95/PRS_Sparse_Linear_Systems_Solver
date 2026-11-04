# Regenerates the build info header on every build, so a benchmark row can never
# carry the commit from an older configure. The file only changes when the commit does.
#
# Inputs: SOURCE_DIR, OUTPUT, FLAGS.
find_program(GIT_EXECUTABLE git)
set(commit "unknown")
if(GIT_EXECUTABLE)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -c safe.directory=* -C "${SOURCE_DIR}" rev-parse --short=12 HEAD
    OUTPUT_VARIABLE head OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE head_result)
  if(head_result EQUAL 0)
    set(commit "${head}")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -c safe.directory=* -C "${SOURCE_DIR}" status --porcelain --ignore-submodules=dirty
      OUTPUT_VARIABLE changes OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(changes)
      string(APPEND commit "-dirty")
    endif()
  endif()
endif()

set(content "#define SSOLVE_GIT_COMMIT \"${commit}\"\n#define SSOLVE_BUILD_FLAGS \"${FLAGS}\"\n")
set(previous "")
if(EXISTS "${OUTPUT}")
  file(READ "${OUTPUT}" previous)
endif()
if(NOT previous STREQUAL content)
  file(WRITE "${OUTPUT}" "${content}")
endif()
