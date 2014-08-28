# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)


# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)


# Set variable for current working directory (workspace directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/PetaVision CACHE FILEPATH "The base petavision directory")

#Set CMAKE_MODULE_PATH
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PV_DIR}/docs/cmake/Modules/")
set(CMAKE_MODULE_PATH "${PV_DIR}/docs/cmake/Modules/")

# Set this as a compiler directive for the code
#add_Definitions(-DPV_DIR="${PV_DIR}")

set(OPEN_MP_THREADS FALSE CACHE PATH "Defines if PetaVision uses OpenMP threads")
set(OPEN_CL_GPU FALSE CACHE PATH "Defines if PetaVision uses OpenCL GPU")
set(CUDA_GPU FALSE CACHE PATH "Defines if PetaVision uses OpenCL GPU")
set(LIB_SND_FILE FALSE CACHE PATH "Defines if PetaVision uses LIB_SND_FILE")
set(CUDA_RELEASE FALSE CACHE PATH "Defines if Cuda compiles with optimization")

set(OpenCL_dir /usr/local/cuda-6.0/ CACHE PATH "Location of opencl headers")
#set(Cuda_lib /usr/local/cuda-6.0/lib64 CACHE PATH "Location of opencl headers")

if(OPEN_MP_THREADS)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DPV_USE_THREADS")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -DPV_USE_THREADS")
endif (OPEN_MP_THREADS)

if(OPEN_CL_GPU)
   IF(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework opencl -DPV_OPENCL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -framework opencl -DPV_OPENCL")
   ELSE(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL -DPV_OPENCL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL -DPV_OPENCL")
   ENDIF(APPLE)
endif (OPEN_CL_GPU)

if(CUDA_GPU)
    #Set include file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE} -DPV_CUDA")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE} -DPV_CUDA")
endif(CUDA_GPU)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPV_DIR=\"\\\"${PV_DIR}\\\"\"")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPV_DIR=\"${PV_DIR}\"")


if (LIB_SND_FILE)
    find_package(LIBSNDFILE)
endif (LIB_SND_FILE)

if(LIB_SND_FILE)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPV_USE_SNDFILE")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPV_USE_SNDFILE")
endif (LIB_SND_FILE)

# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
#    

set(MPI_C_COMPILER_NAME "${CMAKE_C_COMPILER}")
set(MPI_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}")

add_subdirectory(${PV_DIR})
# # Insert additional add_subdirectory commands to add other projects.
#add_subdirectory(BIDS)
#add_subdirectory(DepthLCA)
#add_subdirectory(HyPerHLCA)
#add_subdirectory(MaskLCA)
#add_subdirectory(textAnalysis)
#add_subdirectory(soundAnalysis)

add_subdirectory(PVSystemTests)
