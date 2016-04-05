# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)

set(PVMATCHINGPURSUIT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src" CACHE PATH "The directory path for the PV MatchingPursuit library's header files")
if(NOT EXISTS "${PVMATCHINGPURSUIT_INCLUDE_DIR}")
   message(FATAL_ERROR "PVMATCHINGPURSUIT_INCLUDE_DIR \"${PVMATCHINGPURSUIT_INCLUDE_DIR}\" does not exist.  Please set this variable to the directory that contains the matchingpursuit header files")
endif(NOT EXISTS "${PVMATCHINGPURSUIT_INCLUDE_DIR}")

enable_testing()

# Include the pv_add_test macro
include(PVAddTest)
include(PVAddExecutable)

set(PV_SYSTEM_TEST_THREADS "" CACHE STRING "Defines the number of threads to use for system tests (empty string for max)")

# Make sure that PV_SYSTEM_TEST_THREADS is a number
if("${PV_SYSTEM_TEST_THREADS}" MATCHES "^([0-9]+|)$")
   #Do nothing, correct
else("${PV_SYSTEM_TEST_THREADS}" MATCHES "^([0-9]+|)$")
   message(FATAL_ERROR "PV_SYSTEM_TEST_THREADS must be either a positive integer or the empty string")
endif("${PV_SYSTEM_TEST_THREADS}" MATCHES "^([0-9]+|)$")

include_directories(${PVMATCHINGPURSUIT_INCLUDE_DIR})
link_directories(${PVMATCHINGPURSUIT_LIBRARIES})

add_subdirectory(MatchingPursuitTest)
# # Insert additional add_subdirectory commands to add other projects.
