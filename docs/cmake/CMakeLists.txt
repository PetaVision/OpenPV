# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)


# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)


# Set variable for current working directory (workspace directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/PetaVision CACHE PATH "The base petavision directory")

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${PV_DIR}/docs/cmake/Modules/")

set(OPEN_MP_THREADS FALSE CACHE BOOL "Defines if PetaVision uses OpenMP threads")
set(OPEN_CL_GPU FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
set(CUDA_GPU FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
set(LIB_SND_FILE FALSE CACHE BOOL "Defines if PetaVision uses LIB_SND_FILE")
set(CUDA_RELEASE FALSE CACHE BOOL "Defines if Cuda compiles with optimization")

set(CUDNN FALSE CACHE BOOL "Defines if PetaVision uses Cuda's CUDNN library")
set(CUDNN_PATH /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-R1/ CACHE PATH "Location of cuDNN libraries")

set(OpenCL_dir /usr/local/cuda-6.0/ CACHE PATH "Location of opencl headers")
#set(Cuda_lib /usr/local/cuda-6.0/lib64 CACHE PATH "Location of opencl headers")

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
   set(OPEN_CL_GPU_FLAGS_ADDED TRUE)
endif (OPEN_CL_GPU AND NOT DEFINED OPEN_CL_GPU_FLAGS_ADDED)

if(CUDA_GPU)
    #Set include file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE} -DPV_CUDA")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE} -DPV_CUDA")
    if(CUDNN)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDNN_PATH} -DPV_CUDNN")
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDNN_PATH} -DPV_CUDNN")
    endif(CUDNN)
endif(CUDA_GPU)

if (LIB_SND_FILE AND NOT LIBSNDFILE_FOUND)
    find_package(LIBSNDFILE)
endif (LIB_SND_FILE AND NOT LIBSNDFILE_FOUND)

if(LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)
   include_directories(${LIBSNDFILE_INCLUDE_DIR})
   add_definitions(-DPV_USE_SNDFILE)
   set(LIB_SND_FILE_FLAGS_ADDED TRUE)
endif (LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)

# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
#    

if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
    MESSAGE("-- Searching for MPI library ............")
    find_package(MPI REQUIRED)
endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)

add_subdirectory(${PV_DIR})
# # Insert additional add_subdirectory commands to add other projects.
#add_subdirectory(BIDS)
#add_subdirectory(DepthLCA)
#add_subdirectory(HyPerHLCA)
#add_subdirectory(MaskLCA)
#add_subdirectory(textAnalysis)
#add_subdirectory(soundAnalysis)
