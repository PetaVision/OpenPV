## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)

## Version 0.1.0.0
set(PV_VERSION_MAJOR 0)
set(PV_VERSION_MINOR 1)
set(PV_VERSION_PATCH 0)
set(PV_VERSION_TWEAK 0)

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/docs/cmake/Modules/")

if (NOT CMAKE_BUILD_TYPE)        #To use a different CMAKE_BUILD_TYPE, use the option -DCMAKE_BUILD_TYPE=xxxx on the command line
   set(CMAKE_BUILD_TYPE Release) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel
endif (NOT CMAKE_BUILD_TYPE)

# Variables to set on the command line
set(OPEN_MP_THREADS FALSE CACHE BOOL "Defines if PetaVision uses OpenMP threads")
set(OPEN_CL_GPU FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
set(OpenCL_dir /usr/local/cuda-6.0/ CACHE PATH "Location of opencl headers")
set(CUDA_GPU FALSE CACHE BOOL "Defines if PetaVision uses CUDA GPU")
set(CUDA_RELEASE FALSE CACHE BOOL "Defines if Cuda compiles with optimization")
set(CUDNN FALSE CACHE BOOL "Defines if PetaVision uses Cuda's CUDNN library")
set(CUDNN_PATH /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-R1/ CACHE PATH "Location of cuDNN libraries")
set(PV_BUILD_SHARED FALSE CACHE BOOL "Defines if the pv object is built as a shared object")

set(LIB_SND_FILE FALSE CACHE BOOL "Defines if PetaVision uses LIB_SND_FILE")

IF(APPLE AND CLANG_OMP)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
   set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
ENDIF(APPLE AND CLANG_OMP)

if(OPEN_MP_THREADS AND NOT DEFINED OPEN_MP_THREADS_FLAGS_ADDED)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DPV_USE_THREADS")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -DPV_USE_THREADS")
   set(OPEN_MP_THREADS_FLAGS_ADDED TRUE)
endif (OPEN_MP_THREADS AND NOT DEFINED OPEN_MP_THREADS_FLAGS_ADDED)

if(OPEN_CL_GPU AND NOT DEFINED OPEN_CL_GPU_FLAGS_ADDED)
   IF(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework opencl -DPV_OPENCL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -framework opencl -DPV_OPENCL")
   ELSE(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL -DPV_OPENCL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL -DPV_OPENCL")
   ENDIF(APPLE)
   set(OPEN_CL_GPU_FLAGS_ADDED)
endif (OPEN_CL_GPU AND NOT DEFINED OPEN_CL_GPU_FLAGS_ADDED)

if(OPEN_CL_GPU)
   add_definitions(-DPV_DIR="${CMAKE_CURRENT_SOURCE_DIR}") # needed by HyPerCol.cpp
endif (OPEN_CL_GPU)



set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

if(CUDA_GPU)
    #Set include file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE} -DPV_CUDA")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE} -DPV_CUDA")
    if(CUDNN)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDNN_PATH} -DPV_CUDNN")
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDNN_PATH} -DPV_CUDNN")
    endif(CUDNN)
endif(CUDA_GPU)

# File which denotes compile-ready folders
set(PV_SRC_LIST PV_FOLDER_LIST.txt)

set(PV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PV_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

######################################################################################
######################################################################################
##
## MAIN CODE
##
######################################################################################
######################################################################################

set(LIBRARY_OUTPUT_PATH ${PV_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PV_BINARY_DIR})

# Lex/Yacc for parser
if (NOT BISON_PARSER_DEFINED)
    MESSAGE("-- Searching for bison ..................")
    find_package(BISON REQUIRED)
    BISON_TARGET(PARSER ${PV_SOURCE_DIR}/io/parser/params.y ${PV_SOURCE_DIR}/io/parser-generated/param_parser.cpp COMPILE_FLAGS "-y")
endif (NOT BISON_PARSER_DEFINED)
if (NOT FLEX_SCANNER_DEFINED)
    MESSAGE("-- Searching for flex (yacc) ............")
    find_package(FLEX REQUIRED)
    FLEX_TARGET(SCANNER ${PV_SOURCE_DIR}/io/parser/params.l ${PV_SOURCE_DIR}/io/parser-generated/param_lexer.c)
    ADD_FLEX_BISON_DEPENDENCY(SCANNER PARSER)
endif (NOT FLEX_SCANNER_DEFINED)


# Read in file and parse by line
file(READ "${PV_SRC_LIST}" contents)
STRING(REGEX REPLACE "\n" ";" contents "${contents}")

# Find PetaVision c/cpp files
foreach(LIN ${contents})
   file(GLOB libSrcCPP ${LIN}/*.cpp)
   file(GLOB libSrcC ${LIN}/*.c)
   set(PVLibSrc ${PVLibSrc} ${libSrcCPP})
   set(PVLibSrc ${PVLibSrc} ${libSrcC})

   if(CUDA_GPU)
      file(GLOB_RECURSE libSrcCU ${LIN}/*.cu)
      set(PVLibSrcCu ${PVLibSrcCu} ${libSrcCU})
   endif(CUDA_GPU)
endforeach(LIN ${contents})

# Header file to pass CMake settings to source code
configure_file (
   "cMakeHeader.template"
   "${PV_SOURCE_DIR}/include/cMakeHeader.h"
)

# Add external libraries 
if (NOT GDAL_FOUND)
    MESSAGE("-- Searching for GDAL library ...........")
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)
include_directories(${GDAL_INCLUDE_DIR})

# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
#

if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
    MESSAGE("-- Searching for MPI library ............")
    find_package(MPI REQUIRED)
endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)

set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

if (LIB_SND_FILE AND NOT DEFINED LIBSNDFILE_FOUND)
    MESSAGE("-- Searching for SND library ............")
    find_package(LIBSNDFILE)
endif (LIB_SND_FILE AND NOT DEFINED LIBSNDFILE_FOUND)

if (CUDA_GPU)
    MESSAGE("-- Searching for CUDA library ............")
    find_package(CUDA)
endif (CUDA_GPU)

#Include compiler directive and include directory for LIBSNDFILE
if(LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)
   include_directories(${LIBSNDFILE_INCLUDE_DIR})
   add_definitions(-DPV_USE_SNDFILE)
   set(LIB_SND_FILE_FLAGS_ADDED)
endif (LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)

#IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
#   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g3; -G3; -O0)
#ELSE(CMAKE_BUILD_TYPE MATCHES RELEASE)
#   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g0; -G0; -O3)
#endif(CMAKE_BUILD_TYPE)

if(CUDNN)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_30; )
else(CUDNN)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_20;)
endif(CUDNN)


if(CUDA_RELEASE)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O;)
else(CUDA_RELEASE)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xptxas; -v; -keep; -lineinfo; -g; -G;)
endif(CUDA_RELEASE)
#cuda_compile(CUDA_O ${PVLibSrcCu}) 

# Add PetaVision
if(CUDA_GPU)
   if(PV_BUILD_SHARED)
      cuda_add_library(pv SHARED ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS} ${PVLibSrcCu})
   else(PV_BUILD_SHARED)
      cuda_add_library(pv STATIC ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS} ${PVLibSrcCu})
   endif(PV_BUILD_SHARED)
else(CUDA_GPU)
   if(PV_BUILD_SHARED)
      add_library(pv SHARED ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS})
   else(PV_BUILD_SHARED)
      add_library(pv STATIC ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS})
   endif(PV_BUILD_SHARED)
endif(CUDA_GPU)

# Link executable to external libraries
target_link_libraries(pv ${GDAL_LIBRARY})
target_link_libraries(pv ${MPI_LIBRARY})
IF(LIBSNDFILE_FOUND)
   target_link_libraries(pv ${LIBSNDFILE_LIBRARY})
ENDIF(LIBSNDFILE_FOUND)

IF(CUDNN)
   IF(APPLE)
      target_link_libraries(pv ${CUDNN_PATH}/libcudnn.dylib)
   ELSE(APPLE)
      target_link_libraries(pv ${CUDNN_PATH}/libcudnn.so)
   ENDIF(APPLE)
ENDIF(CUDNN)


