#
# Adds an OpenPV library. Takes a target name and a list of files
# to build the target
#
# Exepcted in scope variables to be set:
#
# PV_CONFIG_FILE_DIR
# PV_INCLUDE_DIR: path to OpenPV include files
# PV_LIBRARIES: full path to libpv.a, libpv.so or libpv.dylib
#
# Optional in scope variables 
#
# GDAL_FOUND, GDAL_INCLUDE_DIR and GDAL_LIBRARIES. These can be set with find_package(GDAL)
# MPI_FOUND, MPI_CXX_INCLUDE_PATH and MPI_CXX_LIBRARIES. Can be set with find_package(MPI)
# CUDA_FOUND. Can be set with find_package(CUDA). Requires CUDNN
# CUDNN_FOUND, CUDNN_INCLUDE_DIR, CUDNN_LIBRARIES. Can be set with find_package(CUDNN).
#  FindCUDNN.cmake is provided in ${PV_SOURCE_DIR}/OpenPV/cmake
#

macro(pv_add_library TARGET)
  cmake_parse_arguments(
    PARSED_ARGS
    ""
    "OUTPUT_PATH"
    "SRC"
    ${ARGN}
    )

  if (PV_BUILD_SHARED) 
    message(STATUS "Building shared library for ${TARGET}")
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
    cuda_add_library(${TARGET} ${SHARED_FLAG} ${PARSED_ARGS_SRC})
  else()
    # Add library target, no CUDA
    STRING(REGEX REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_library(${TARGET} ${SHARED_FLAG} ${PARSED_ARGS_SRC})
  endif()
endmacro()