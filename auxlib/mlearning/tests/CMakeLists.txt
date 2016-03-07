include(PVAddTest)
include(PVAddExecutable)

enable_testing()

set(PVMLEARNING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../" CACHE PATH "The base directory for the PV mlearning library")

if(NOT EXISTS "${PVMLEARNING_SOURCE_DIR}")
   message(FATAL_ERROR "PVMLEARNING_SOURCE_DIR \"${PVMLEARNING_SOURCE_DIR}\" does not exist.")
endif()

set(PVMLEARNING_INCLUDE_DIR "${PVMLEARNING_SOURCE_DIR}/src")

include_directories(${PVMLEARNING_INCLUDE_DIR})

add_subdirectory(MLPTest)
