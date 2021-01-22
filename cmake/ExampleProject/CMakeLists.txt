# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set

cmake_minimum_required(VERSION 3.0.0)

get_filename_component(PV_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PV_PROJECT_NAME})

set(PV_SOURCE_DIR "$ENV{HOME}/OpenPV" CACHE PATH "Path to the OpenPV repository")

if (NOT EXISTS "${PV_SOURCE_DIR}" OR NOT IS_DIRECTORY "${PV_SOURCE_DIR}")
   message(FATAL_ERROR "\"${PV_SOURCE_DIR}\" does not exist.\nMake sure that PV_SOURCE_DIR points to the PetaVision OpenPV repository")
endif (NOT EXISTS "${PV_SOURCE_DIR}" OR NOT IS_DIRECTORY "${PV_SOURCE_DIR}")

if (NOT EXISTS "${PV_SOURCE_DIR}/cmake" OR NOT IS_DIRECTORY "${PV_SOURCE_DIR}/cmake")
   message(FATAL_ERROR "PV_SOURCE_DIR=\"${PV_SOURCE_DIR} does not have a cmake directory.\nMake sure that PV_SOURCE_DIR points to the OpenPV repository")
endif (NOT EXISTS "${PV_SOURCE_DIR}/cmake" OR NOT IS_DIRECTORY "${PV_SOURCE_DIR}/cmake")

set(CMAKE_MODULE_PATH "${PV_SOURCE_DIR}/cmake")

set(PV_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/OpenPV")

add_subdirectory(${PV_SOURCE_DIR} ${PV_BUILD_DIR})

pv_config_project()

set(${PV_PROJECT_NAME}_SRCFILES
   src/main.cpp
)

set(PV_CONFIG_FILE_DIR "${PV_BUILD_DIR}/lib/include")
set(PV_INCLUDE_DIR "${PV_SOURCE_DIR}")
set(PV_LIBRARIES "${PV_BUILD_DIR}/lib/libpv.a")

pv_add_executable(
   ${PV_PROJECT_NAME}
   SRC ${${PV_PROJECT_NAME}_SRCFILES}
   OUTPUT_PATH  "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}"
)

add_dependencies(${PV_PROJECT_NAME} pv)

set(SOURCE_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/input")
if (EXISTS "${SOURCE_INPUT_DIR}")
  set(BUILD_INPUT_LINK "${CMAKE_CURRENT_BINARY_DIR}/input")
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SOURCE_INPUT_DIR} ${BUILD_INPUT_LINK})
endif (EXISTS "${SOURCE_INPUT_DIR}")
