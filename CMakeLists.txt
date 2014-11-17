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

set(OPEN_MP_THREADS FALSE CACHE PATH "Defines if PetaVision uses OpenMP threads")
set(OPEN_CL_GPU FALSE CACHE PATH "Defines if PetaVision uses OpenCL GPU")
set(CUDA_GPU FALSE CACHE PATH "Defines if PetaVision uses OpenCL GPU")

if(OPEN_MP_THREADS)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DPV_USE_THREADS")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -DPV_USE_THREADS")
endif (OPEN_MP_THREADS)

if(OPEN_CL_GPU)
   #TODO: make sure these flags work on linux
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPV_OPENCL")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPV_OPENCL")
endif (OPEN_CL_GPU)

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")


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
    BISON_TARGET(PARSER ${PV_SOURCE_DIR}/io/parser/params.y ${PV_SOURCE_DIR}/io/parser/param_parser.cpp COMPILE_FLAGS "-y")
endif (NOT BISON_PARSER_DEFINED)
if (NOT FLEX_SCANNER_DEFINED)
    MESSAGE("-- Searching for flex (yacc) ............")
    find_package(FLEX REQUIRED)
    FLEX_TARGET(SCANNER ${PV_SOURCE_DIR}/io/parser/params.l ${PV_SOURCE_DIR}/io/parser/param_lexer.c)
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

if (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
    MESSAGE("-- Searching for MPI library ............")
    find_package(MPI REQUIRED)
    set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
    set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
endif (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)

if (LIB_SND_FILE)
    MESSAGE("-- Searching for SND library ............")
    find_package(LIBSNDFILE)
endif (LIB_SND_FILE)

if (CUDA_GPU)
    MESSAGE("-- Searching for CUDA library ............")
    find_package(CUDA)
endif (CUDA_GPU)

#Include compiler directive and include directory for LIBSNDFILE
if (LIB_SND_FILE)
   ADD_DEFINITIONS(-DPV_USE_SNDFILE)
   include_directories(${LIBSNDFILE_INCLUDE_DIR})
else (LIB_SND_FILE)
   REMOVE_DEFINITIONS(-DPV_USE_SNDFILE)
endif (LIB_SND_FILE)

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
   cuda_add_library(pv ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS} ${PVLibSrcCu})
else(CUDA_GPU)
   add_library(pv ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS})
endif(CUDA_GPU)


