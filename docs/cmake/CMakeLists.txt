# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)


# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)


# Set variable for current working directory (workspace directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/PetaVision CACHE PATH "The base petavision directory")


########
# March 31, 2015: renamed several cache variables to agree with C/C++ preprocessor directives
# so that they could be put into cMakeHeader.h easily instead of passing -D arguments to the compile command.
# Accordingly, test if old variable name exists, and if so, use its value as the new variable.
# then delete the old variable from the cache.  After a while, no one
# should have the old variables in the cache and this section can be deleted.
if (DEFINED OPEN_MP_THREADS)
   set(PV_USE_THREADS ${OPEN_MP_THREADS} CACHE BOOL "Defines whether PetaVision uses OpenMP threads")
   unset(OPEN_MP_THREADS CACHE)
endif (DEFINED OPEN_MP_THREADS)
if (DEFINED OPEN_CL_GPU)
   set(PV_OPENCL ${OPEN_CL_GPU} CACHE BOOL "Defines whether PetaVision uses OpenCL GPU")
   unset(OPEN_CL_GPU CACHE)
endif (DEFINED OPEN_CL_GPU)
if (DEFINED CUDA_GPU)
   set(PV_CUDA ${CUDA_GPU} CACHE BOOL "Defines whether PetaVision uses CUDA GPU")
   unset(CUDA_GPU CACHE)
endif (DEFINED CUDA_GPU)
if (DEFINED CUDNN)
   set(PV_CUDA ${CUDNN} CACHE BOOL "Defines whether PetaVision uses CUDA's cuDNN library")
   unset(CUDNN CACHE)
endif (DEFINED CUDNN)
########


#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${PV_DIR}/docs/cmake/Modules/")

set(PV_USE_THREADS TRUE CACHE BOOL "Defines if PetaVision uses OpenMP threads")
set(PV_OPENCL FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
set(PV_CUDA FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
# Sound-related classes moved to auxlib/pvsound Mar 16, 2015
# set(LIB_SND_FILE FALSE CACHE BOOL "Defines if PetaVision uses LIB_SND_FILE")
set(CUDA_RELEASE FALSE CACHE BOOL "Defines if Cuda compiles with optimization")

set(PV_CUDNN FALSE CACHE BOOL "Defines if PetaVision uses Cuda's CUDNN library")
set(CUDNN_PATH /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/ CACHE PATH "Location of cuDNN libraries")

set(CLANG_OMP TRUE CACHE BOOL "Defines if we're compiling with Clang + omp support")

set(OpenCL_dir /usr/local/cuda-6.5/ CACHE PATH "Location of opencl headers")

if(PV_USE_THREADS AND NOT DEFINED PV_USE_THREADS_ADDED)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
   set(PV_USE_THREADS_ADDED TRUE)
endif (PV_USE_THREADS AND NOT DEFINED PV_USE_THREADS_ADDED)

if(PV_OPENCL AND NOT DEFINED PV_OPENCL_FLAGS_ADDED)
   IF(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework opencl")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -framework opencl")
   ELSE(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL")
   ENDIF(APPLE)
   set(PV_OPENCL_FLAGS_ADDED TRUE)
endif (PV_OPENCL AND NOT DEFINED PV_OPENCL_FLAGS_ADDED)

if(PV_CUDA)
    #Set include file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE}")
    if(PV_CUDNN)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDNN_PATH}")
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDNN_PATH}")
    endif(PV_CUDNN)
endif(PV_CUDA)

# Sound-related classes moved to auxlib/pvsound Mar 16, 2015 
# if (LIB_SND_FILE AND NOT LIBSNDFILE_FOUND)
#     find_package(LIBSNDFILE)
# endif (LIB_SND_FILE AND NOT LIBSNDFILE_FOUND)
# 
# if(LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)
#    include_directories(${LIBSNDFILE_INCLUDE_DIR})
#    add_definitions(-DPV_USE_SNDFILE)
#    set(LIB_SND_FILE_FLAGS_ADDED TRUE)
# endif (LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)

# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
#    

if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
    MESSAGE("-- Searching for MPI library ............")
    find_package(MPI REQUIRED)
endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)

add_subdirectory(${PV_DIR})
add_subdirectory(PVSystemTests)
# # Insert additional add_subdirectory commands to add other projects.
#add_subdirectory(BIDS)
#add_subdirectory(DepthLCA)
#add_subdirectory(HyPerHLCA)
#add_subdirectory(MaskLCA)
#add_subdirectory(textAnalysis)
#add_subdirectory(soundAnalysis)
