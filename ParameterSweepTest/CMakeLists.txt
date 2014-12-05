## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PV_DIR}/docs/cmake/Modules/")

string(FIND ${CMAKE_CURRENT_SOURCE_DIR} "/" pos REVERSE)
MATH(EXPR pos ${pos}+1)
string(SUBSTRING ${CMAKE_CURRENT_SOURCE_DIR} ${pos} -1 PV_PROJECT_NAME)

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel
endif (NOT CMAKE_BUILD_TYPE)

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

set(PV_SOURCE_DIR "${PV_DIR}/src")
set(PV_BINARY_DIR "${PV_DIR}/lib")
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if (NOT GDAL_FOUND)
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)

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

if (LIB_SND_FILE AND NOT LIBSNDFILE_FOUND)
    find_package(LIBSNDFILE)
endif (LIB_SND_FILE AND NOT LIBSNDFILE_FOUND)

if(LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)
   include_directories(${LIBSNDFILE_INCLUDE_DIR})
   add_definitions(-DPV_USE_SNDFILE)
   set(LIB_SND_FILE_FLAGS_ADDED TRUE)
endif (LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)

include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})

# Link to binary files
link_directories(${PV_BINARY_DIR})

# Add executable
file(GLOB libSrcCPP ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB libSrcC ${PROJECT_SOURCE_DIR}/*.c)

if(CUDA_GPU)
   cuda_add_executable(${PV_PROJECT_NAME} ${libSrcCPP} ${libSrcC})
else(CUDA_GPU)
   add_executable(${PV_PROJECT_NAME} ${libSrcCPP} ${libSrcC})
endif(CUDA_GPU)

target_link_libraries(${PV_PROJECT_NAME} pv)

# Link executable to external libraries
target_link_libraries(${PV_PROJECT_NAME} ${GDAL_LIBRARY})
IF(LIBSNDFILE_FOUND)
   target_link_libraries(${PV_PROJECT_NAME} ${LIBSNDFILE_LIBRARY})
ENDIF(LIBSNDFILE_FOUND)

IF(CUDNN)
   target_link_libraries(${PV_PROJECT_NAME} ${CUDNN_PATH}/libcudnn.so)
ENDIF(CUDNN)

