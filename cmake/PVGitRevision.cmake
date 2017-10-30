# Sets PV_GIT_REVISION
find_package(Git)

if (GIT_FOUND)
   unset(PV_GIT_REVISION)
   # Get current commit
   execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
                   WORKING_DIRECTORY "${SOURCE_DIR}"
                   RESULT_VARIABLE PV_CURRENT_COMMIT_RESULT
                   OUTPUT_VARIABLE PV_CURRENT_COMMIT
                   ERROR_VARIABLE PV_CURRENT_COMMIT_ERROR
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
   if (${PV_CURRENT_COMMIT_RESULT} EQUAL 0)
      # Get commit hash and date commit was authored
      execute_process(COMMAND "${GIT_EXECUTABLE}" log -n 1 "--format=%H (%ad)" "${PV_CURRENT_COMMIT}"
                      WORKING_DIRECTORY "${SOURCE_DIR}"
                      RESULT_VARIABLE PV_GIT_REVISION_RESULT
                      OUTPUT_VARIABLE PV_GIT_REVISION
                      ERROR_VARIABLE PV_GIT_REVISION_ERROR
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(PV_GIT_REVISION "git repository version ${PV_GIT_REVISION}")
      # See if there are any local changes
      execute_process(COMMAND "${GIT_EXECUTABLE}" status -s
                      WORKING_DIRECTORY "${SOURCE_DIR}"
                      RESULT_VARIABLE PV_GIT_STATUS_RESULT
                      OUTPUT_VARIABLE PV_GIT_STATUS_OUTPUT
                      ERROR_VARIABLE VP_GIT_STATUS_ERROR
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(LENGTH "${PV_GIT_STATUS_OUTPUT}" PV_STATUS_LENGTH)
      if(NOT ("${PV_STATUS_LENGTH}" EQUAL 0))
         set(PV_GIT_REVISION "${PV_GIT_REVISION} with local modifications")
      endif(NOT ("${PV_STATUS_LENGTH}" EQUAL 0))
   else (${PV_CURRENT_COMMIT_RESULT} EQUAL 0)
      unset(PV_GIT_REVISION)
   endif (${PV_CURRENT_COMMIT_RESULT} EQUAL 0)
endif (GIT_FOUND)

if (NOT DEFINED PV_GIT_REVISION)
   set(PV_GIT_REVISION "unknown version")
endif ()

file(MAKE_DIRECTORY ${PV_CONFIG_FILE_DIR})
configure_file (
   "${SOURCE_DIR}/src/pvGitRevision.template"
   "${PV_CONFIG_FILE_DIR}/pvGitRevision.h"
)

