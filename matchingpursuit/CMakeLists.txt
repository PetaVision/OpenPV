## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)

string(FIND ${CMAKE_CURRENT_SOURCE_DIR} "/" pos REVERSE)
MATH(EXPR pos ${pos}+1)
string(SUBSTRING ${CMAKE_CURRENT_SOURCE_DIR} ${pos} -1 PV_AUXLIB_NAME)

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/Modules/")

## Version 0.1.0.0
set(PV_VERSION_MAJOR 0)
set(PV_VERSION_MINOR 1)
set(PV_VERSION_PATCH 0)
set(PV_VERSION_TWEAK 0)

# Variables to set on the command line
set(OPEN_MP_THREADS FALSE CACHE BOOL "Defines if PetaVision uses OpenMP threads")
set(OPEN_CL_GPU FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
set(OpenCL_dir /usr/local/cuda-6.0/ CACHE PATH "Location of opencl headers")
set(CUDA_GPU FALSE CACHE BOOL "Defines if PetaVision uses CUDA GPU")
set(CUDA_RELEASE FALSE CACHE BOOL "Defines if Cuda compiles with optimization")
set(CUDNN FALSE CACHE BOOL "Defines if PetaVision uses Cuda's CUDNN library")
set(CUDNN_PATH /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-R1/ CACHE PATH "Location of cuDNN libraries")
set(PV_BUILD_SHARED FALSE CACHE BOOL "Defines if the pv object is built as a shared object")
set(CLANG_OMP FALSE CACHE BOOL "Defines if we're compiling with Clang + omp support")
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../trunk CACHE PATH "The base PetaVision directory")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PV_DIR}/docs/cmake/Modules/")

IF(CUDA_GPU AND OPEN_CL_GPU)
   message(FATAL_ERROR "CUDA_GPU and OPEN_CL_GPU cannot both be true")
ENDIF(CUDA_GPU AND OPEN_CL_GPU)

IF(NOT EXISTS "${PV_DIR}")
   message(FATAL_ERROR "PV_DIR \"${PV_DIR}\" does not exist.")
ENDIF(NOT EXISTS "${PV_DIR}")

IF(APPLE AND CLANG_OMP AND NOT DEFINED APPLE_CLANG_OMP_FLAGS_ADDED)
   message(WARNING "Using OpenMP threads with Clang. Make sure that your version of Clang supports OpenMP.")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
   set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
   set(APPLE_CLANG_OMP_FLAGS_ADDED TRUE)
ENDIF(APPLE AND CLANG_OMP AND NOT DEFINED APPLE_CLANG_OMP_FLAGS_ADDED)

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

if(CUDA_GPU AND NOT DEFINED CUDA_GPU_FLAGS_ADDED)
    #Set include file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE} -DPV_CUDA")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE} -DPV_CUDA")
    if(CUDNN)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDNN_PATH} -DPV_CUDNN")
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDNN_PATH} -DPV_CUDNN")
    endif(CUDNN)
    set(CUDA_GPU_FLAGS_ADDED)
endif(CUDA_GPU AND NOT DEFINED CUDA_GPU_FLAGS_ADDED)

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

set(PV_AUXLIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PV_AUXLIB_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

######################################################################################
######################################################################################
##
## MAIN CODE
##
######################################################################################
######################################################################################

set(LIBRARY_OUTPUT_PATH ${PV_AUXLIB_LIB_DIR})

file(GLOB libSrcCPP ${PV_AUXLIB_SRC_DIR}/*.cpp)
set(PVauxlibSrc ${PVauxlibSrc} ${libSrcCPP})
file(GLOB libSrcC ${PV_AUXLIB_SRC_DIR}/*.c)
set(PVauxlibSrc ${PVauxlibSrc} ${libSrcC})
if(CUDA_GPU)
   file(GLOB libSrcCU ${PV_AUXLIB_SRC_DIR}/*.cu)
   set(PVauxlibSrcCu ${PVauxlibSrcCu} ${libSrcCU})
endif(CUDA_GPU)

# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
#

if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
    MESSAGE("-- Searching for MPI library ............")
    find_package(MPI REQUIRED)
endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
if (NOT GDAL_FOUND)
    MESSAGE("-- Searching for GDAL library ...........")
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)

set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

if (CUDA_GPU)
    MESSAGE("-- Searching for CUDA library ............")
    find_package(CUDA)
endif (CUDA_GPU)

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
#cuda_compile(CUDA_O ${PVauxlibSrcCu}) 

# Add PetaVision
include_directories(${PV_DIR}/src)
link_directories(${PV_DIR}/lib)

if(CUDA_GPU)
   if(PV_BUILD_SHARED)
      cuda_add_library(${PV_AUXLIB_NAME} SHARED ${PVauxlibSrc} ${PVauxlibSrcCu})
   else(PV_BUILD_SHARED)
      cuda_add_library(${PV_AUXLIB_NAME} STATIC ${PVauxlibSrc} ${PVauxlibSrcCu})
   endif(PV_BUILD_SHARED)
else(CUDA_GPU)
   if(PV_BUILD_SHARED)
      add_library(${PV_AUXLIB_NAME} SHARED ${PVauxlibSrc})
   else(PV_BUILD_SHARED)
      add_library(${PV_AUXLIB_NAME} STATIC ${PVauxlibSrc})
   endif(PV_BUILD_SHARED)
endif(CUDA_GPU)

# Link executable to external libraries
target_link_libraries(${PV_AUXLIB_NAME} pv)
target_link_libraries(${PV_AUXLIB_NAME} ${GDAL_LIBRARY})
target_link_libraries(${PV_AUXLIB_NAME} ${MPI_LIBRARY})

IF(CUDNN)
   IF(APPLE)
      target_link_libraries(${PV_AUXLIB_NAME} ${CUDNN_PATH}/libcudnn.dylib)
   ELSE(APPLE)
      target_link_libraries(${PV_AUXLIB_NAME} ${CUDNN_PATH}/libcudnn.so)
   ENDIF(APPLE)
ENDIF(CUDNN)

