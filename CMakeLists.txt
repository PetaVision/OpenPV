# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)

project(PetaVision)

################################################################################
# Default values
################################################################################

# Default PetaVision core library directory
set(PV_DIR_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PV_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(PV_DEMOS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/demos")
set(PV_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tools")

# Defaults
set(PV_BUILD_DEMOS_DEFAULT OFF)
set(PV_BUILD_TOOLS_DEFAULT OFF)

# Help strings
set(PV_BUILD_DEMOS_HELP "Build OpenPV demos")
set(PV_BUILD_DEMOS_HELP "Build OpenPV tools")

# Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(PV_BUILD_DEMOS ${PV_BUILD_DEMOS_DEFAULT} CACHE BOOL "${PV_BUILD_DEMOS_HELP}")
set(PV_BUILD_TOOLS ${PV_BUILD_TOOLS_DEFAULT} CACHE BOOL "${PV_BUILD_TOOLS_HELP}")

include(PVConfigProject)
pv_config_project()

################################################################################
# Add subdirectories
################################################################################

# Adds the core PetaVision directory
add_subdirectory(${PV_DIR})

if (${PV_BUILD_TEST})
  add_subdirectory(${PV_TEST_DIR})
else()
   message(STATUS "Only building the OpenPV library. Set PV_BUILD_TEST=On to build the OpenPV test suite.")
endif()

if (${PV_BUILD_DEMOS})
   add_subdirectory(${PV_DEMOS_DIR})
endif()

if (${PV_BUILD_TOOLS})
   add_subdirectory(${PV_TOOLS_DIR})
endif()
