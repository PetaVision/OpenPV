## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)

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

if(PV_USE_OPENCL AND NOT DEFINED PV_USE_OPENCL_FLAGS_ADDED)
   IF(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework opencl")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -framework opencl")
   ELSE(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL")
   ENDIF(APPLE)
   set(PV_USE_OPENCL_FLAGS_ADDED TRUE)
endif (PV_USE_OPENCL AND NOT DEFINED PV_USE_OPENCL_FLAGS_ADDED)

set(PV_SOURCE_DIR "${PV_DIR}/src")
set(PV_BINARY_DIR "${PV_DIR}/lib")
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")

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
#
## target_link_libraries commands no longer needed because the libraries are included in libpv.so
## Link executable to external libraries
#target_link_libraries(${PV_PROJECT_NAME} ${GDAL_LIBRARY})
#
#IF(PV_USE_CUDNN)
#   target_link_libraries(${PV_PROJECT_NAME} ${CUDNN_PATH}/libcudnn.so)
#ENDIF(PV_USE_CUDNN)
#
