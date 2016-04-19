#
# Adds an OpenPV based executable. Takes a target name and a list of files
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

macro(pv_add_executable TARGET)
  cmake_parse_arguments(
    PARSED_ARGS
    ""
    "OUTPUT_PATH"
    "SRC"
    ${ARGN}
    )

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

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PV_CPP_11X_FLAGS}")

  if(PV_USE_CUDA AND CUDA_FOUND AND CUDNN_FOUND)
    #list(APPEND CMAKE_CXX_FLAGS ${PV_NVCC_FLAGS})
    STRING(REGEX REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    # It's hard to believe that this works. It's expected that we need
    # cuda_add_executable to get a cuda enabled executable.
    # This was changed for c++11 support. 
    add_executable(${TARGET} ${PARSED_ARGS_SRC})
    target_link_libraries(${TARGET} ${CUDNN_LIBRARIES})
  else()
    # Add  PetaVision library target, no CUDA
    STRING(REGEX REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_executable(${TARGET} ${PARSED_ARGS_SRC})
  endif()

  target_link_libraries(${TARGET} ${PV_LIBRARIES})

  # Set target properties
  if(PARSED_ARGS_OUTPUT_PATH)
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PARSED_ARGS_OUTPUT_PATH})
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PARSED_ARGS_OUTPUT_PATH})
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PARSED_ARGS_OUTPUT_PATH})
  endif()

  if (GDAL_FOUND AND PV_USE_GDAL)
    target_link_libraries(${TARGET} ${GDAL_LIBRARY})
  endif()

  if (PV_USE_MPI AND MPI_FOUND)
    target_link_libraries(${TARGET} ${MPI_CXX_LIBRARIES})
  endif()

  if (MPI_FOUND AND PV_USE_MPI)
    target_link_libraries(${TARGET} ${MPI_CXX_LIBRARIES})
  endif()

  if (PV_USE_OPENMP)
    target_link_libraries(${TARGET} ${PV_OPENMP_LIBRARIES})
  endif()

  # This looks redundant, but linking order of cuda libraries can make a difference. Including
  # these a second time is a bit of a hack, but it can fix things in some cases
  if (PV_USE_CUDA)
    target_link_libraries(${TARGET} ${CUDA_LIBRARIES})
    target_link_libraries(${TARGET} ${CUDNN_LIBRARIES})
  endif()
endmacro()

