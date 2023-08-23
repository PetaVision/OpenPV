## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 3.7.0)

set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../pv-core CACHE FILEPATH
"The directory for the PetaVision library")

IF(NOT EXISTS "${PV_DIR}")
   message(FATAL_ERROR "PV_DIR \"${PV_DIR}\" does not exist.")
ENDIF(NOT EXISTS "${PV_DIR}")

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PV_DIR}/docs/cmake/Modules/")

string(FIND ${CMAKE_CURRENT_SOURCE_DIR} "/" pos REVERSE)
MATH(EXPR pos ${pos}+1)
string(SUBSTRING ${CMAKE_CURRENT_SOURCE_DIR} ${pos} -1 PV_PROJECT_NAME)

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

set(BUILD_DIR "" CACHE STRING "The directory to write the executable binary into. (if blank, CMAKE_BUILD_TYPE will give the build directory")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel
endif (NOT CMAKE_BUILD_TYPE)

if(PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)
   if(APPLE AND CLANG_OMP)
      message(WARNING "Using OpenMP threads with Clang. Make sure that your version of Clang supports OpenMP.")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -stdlib=libstdc++")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
   endif(APPLE AND CLANG_OMP)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PV_OPENMP_COMPILER_FLAG}")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PV_OPENMP_COMPILER_FLAG}")
   set(PV_USE_OPENMP_THREADS_FLAGS_ADDED TRUE)
endif (PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)

set(PV_SOURCE_DIR "${PV_DIR}/src")
set(PV_BINARY_DIR "${PV_DIR}/lib")
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")

# Define the directory for the executable file.
# If BUILD_DIR is empty, use CMAKE_BUILD_TYPE string as the name.
# If BUILD_DIR starts with "/", use it as an absolute path
# If BUILD_DIR is nonempty and doesn't start with "/", use it as a path
# relative to the current directory
if("${BUILD_DIR}" STREQUAL "")
   set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")
else("${BUILD_DIR}" STREQUAL "")
   string(SUBSTRING "${BUILD_DIR}" 0 1 BUILD_DIR_FIRSTCHAR)
   if("${BUILD_DIR_FIRSTCHAR}" STREQUAL "/")
      set(PROJECT_BINARY_DIR "${BUILD_DIR}")
   else("${BUILD_DIR_FIRSTCHAR}" STREQUAL "/")
      set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${BUILD_DIR}")
   endif("${BUILD_DIR_FIRSTCHAR}" STREQUAL "/")
endif("${BUILD_DIR}" STREQUAL "")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if (NOT GDAL_FOUND)
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)

if (PV_USE_MPI)
    # The user may specify a non-standard compiler name for MPI
    #  For example, the user may type:
    #    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
    #

    set(MPI_C_COMPILER_NAME "${CMAKE_C_COMPILER}")
    set(MPI_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}")

    if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
        find_package(MPI REQUIRED)
    endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
    set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
    set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
endif (PV_USE_MPI)

if(PV_USE_CUDA)
    #Set include file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE}")
    if(PV_USE_CUDNN)
       file(GLOB_RECURSE CUDNN_HEADER "${CUDNN_PATH}*/cudnn.h")
       string(REGEX REPLACE "/[^/]+$" "" CUDNN_INCLUDE_PATH "${CUDNN_HEADER}")
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDNN_INCLUDE_PATH}")
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDNN_INCLUDE_PATH}")
    endif(PV_USE_CUDNN)
endif(PV_USE_CUDA)

include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})

# Link to binary files
link_directories(${PV_BINARY_DIR})

# Add executable
file(GLOB libSrcCPP ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB libSrcC ${PROJECT_SOURCE_DIR}/*.c)

if(PV_USE_CUDA)
   cuda_add_executable(${PV_PROJECT_NAME} ${libSrcCPP} ${libSrcC})
else(PV_USE_CUDA)
   add_executable(${PV_PROJECT_NAME} ${libSrcCPP} ${libSrcC})
endif(PV_USE_CUDA)

target_link_libraries(${PV_PROJECT_NAME} pv)

# still needed on Mac if pv is built as a shared object library
target_link_libraries(${PV_PROJECT_NAME} ${GDAL_LIBRARY})

## target_link_libraries command no longer needed because the libraries are included in libpv.so
#
#IF(PV_USE_CUDNN)
#   IF(APPLE)
#      file(GLOB_RECURSE CUDNN_LIBRARY "${CUDNN_PATH}*/libcudnn.dylib")
#   ELSE(APPLE)
#      file(GLOB_RECURSE CUDNN_LIBRARY "${CUDNN_PATH}*/libcudnn.so")
#   ENDIF(APPLE)
#   target_link_libraries(pv ${CUDNN_LIBRARY})
#ENDIF(PV_USE_CUDNN)

