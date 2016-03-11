## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)

## Version 0.1.0.0
set(PV_VERSION_MAJOR 0)
set(PV_VERSION_MINOR 1)
set(PV_VERSION_PATCH 0)
set(PV_VERSION_TWEAK 0)

# Nicer for in-source builds. Puts the library into a subdir
set(LIBRARY_OUTPUT_PATH ${PV_LIBRARY_DIR})

set(PV_IO_DIR "${PV_LIBRARY_DIR}/io")
set(PV_PARSER_DIR "${PV_IO_DIR}/parser-generated")
set(PV_PARAM_LEXER_C "${PV_PARSER_DIR}/param_lexer.c")
set(PV_PARAM_PARSER_CPP "${PV_PARSER_DIR}/param_parser.cpp")
set(PV_PARAM_PARSER_HPP "${PV_PARSER_DIR}/param_parser.hpp")

set(PV_LIBRARY_NAME "pv")

# Create directory structure for lexer and parser output
file(MAKE_DIRECTORY ${PV_IO_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${PV_IO_DIR})
file(MAKE_DIRECTORY ${PV_PARSER_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${PV_PARSER_DIR})

# Lex/Yacc for parser
if (NOT BISON_PARSER_DEFINED)
  find_package(BISON REQUIRED)
  BISON_TARGET(PARSER ${PV_SOURCE_DIR}/io/parser/params.y ${PV_PARAM_PARSER_CPP} DEFINES_FILE ${PV_PARAM_PARSER_HPP} COMPILE_FLAGS "-y")
endif (NOT BISON_PARSER_DEFINED)

if (NOT FLEX_SCANNER_DEFINED)
  find_package(FLEX REQUIRED)
  FLEX_TARGET(SCANNER ${PV_SOURCE_DIR}/io/parser/params.l ${PV_PARAM_LEXER_C})
  ADD_FLEX_BISON_DEPENDENCY(SCANNER PARSER)
endif (NOT FLEX_SCANNER_DEFINED)

# Library source subdirectories
set(SRC_SUBDIRS
  src/arch/cuda
  src/arch/mpi
  src/bindings
  src/columns
  src/connections
  src/cudakernels
  src/include
  src/io
  src/kernels
  src/layers
  src/normalizers
  src/utils
  src/weightinit
)

foreach(SUBDIR ${SRC_SUBDIRS})
  include("${SUBDIR}/CMakeLists.txt")
endforeach()

set(PVLibSrc ${PVLibSrcCpp} ${PVLibSrcHpp} ${PVLibSrcC} ${PVLibSrcH})

include(PVGitRevision)

# Header file to pass CMake settings to source code
if(PV_USE_OPENMP)
  # C++ code needs PV_USE_OPENMP_THREADS defined
  set(PV_USE_OPENMP_THREADS ON)
endif()

file(MAKE_DIRECTORY ${PV_CONFIG_FILE_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${PV_CONFIG_FILE_DIR})
configure_file (
   "cMakeHeader.template"
   "${PV_CONFIG_FILE_DIR}/cMakeHeader.h"
)

if (PV_BUILD_SHARED) 
  message(STATUS "Building shared library")
  if (APPLE)
    # Like LD_LIBRARY_PATH, but for OS X
    set(CMAKE_MACOSX_RPATH ${PV_LIBRARY_DIR})
  endif()
  set(SHARED_FLAG "SHARED")
else()
  message(STATUS "Building static library")
  set(SHARED_FLAG "STATIC")
endif()

include_directories(${PV_CONFIG_FILE_DIR})
include_directories(${PV_INCLUDE_DIR})

if (PV_USE_GDAL AND GDAL_FOUND)
  include_directories(${GDAL_INCLUDE_DIR})
endif()

if (PV_USE_CUDA AND CUDNN_FOUND)
  include_directories(${CUDA_TOOLKIT_INCLUDE})
  include_directories(${CUDNN_INCLUDE_DIR})
endif()

if (PV_USE_MPI AND MPI_FOUND)
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinRelSize")
  list(APPEND CMAKE_CXX_FLAGS ${PV_COMPILE_FLAGS_RELEASE})
else()
  list(APPEND CMAKE_CXX_FLAGS ${PV_COMPILE_FLAGS_DEBUG})
endif()

if(PV_USE_CUDA AND CUDA_FOUND AND CUDNN_FOUND)
  list(APPEND CMAKE_CXX_FLAGS ${PV_NVCC_FLAGS})
  STRING(REGEX REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  cuda_add_library(${PV_LIBRARY_NAME} ${SHARED_FLAG} ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS} ${PVLibSrcCu})
else()
  # Add  PetaVision library target, no CUDA
  STRING(REGEX REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  add_library(${PV_LIBRARY_NAME} ${SHARED_FLAG} ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS})
endif()


message(STATUS "cmake build_type: ${CMAKE_BUILD_TYPE}")

# The following is needed to make the Xcode generator put libpv.a into the correct place.
# Without this, Xcode appends a Debug to the end of the output path, regardless of 
# whether or not a Debug build is being created

set_target_properties(${PV_LIBRARY_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PV_LIBRARY_DIR})
set_target_properties(${PV_LIBRARY_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PV_LIBRARY_DIR})
set_target_properties(${PV_LIBRARY_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PV_LIBRARY_DIR})
set_target_properties(${PV_LIBRARY_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PV_LIBRARY_DIR})
set_target_properties(${PV_LIBRARY_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PV_LIBRARY_DIR})
set_target_properties(${PV_LIBRARY_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PV_LIBRARY_DIR})


