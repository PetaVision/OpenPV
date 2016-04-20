include(PVAddTest)
include(PVAddExecutable)

enable_testing()

set(PVMATCHINGPURSUIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../" CACHE PATH "The base directory for the PV matchingpursuit library")

if(NOT EXISTS "${PVMATCHINGPURSUIT_SOURCE_DIR}")
   message(FATAL_ERROR "PVMATCHINGPURSUIT_SOURCE_DIR \"${PVMATCHINGPURSUIT_SOURCE_DIR}\" does not exist.")
endif()

set(PVMATCHINGPURSUIT_INCLUDE_DIR "${PVMATCHINGPURSUIT_SOURCE_DIR}/src")

include_directories(${PVMATCHINGPURSUIT_INCLUDE_DIR})

add_subdirectory(MatchingPursuitTest)
