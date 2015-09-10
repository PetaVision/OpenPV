## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.6)

set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../pv-core CACHE PATH "The base petavision directory")

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PV_DIR}/docs/cmake/Modules/")

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif (NOT CMAKE_BUILD_TYPE)

set(PV_SOURCE_DIR "${PV_DIR}/src")
set(PV_BINARY_DIR "${PV_DIR}/lib")
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if (NOT GDAL_FOUND)
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)

if (PV_USE_MPI)
    if (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
        find_package(MPI REQUIRED)
    endif (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
    set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
    set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
endif (PV_USE_MPI)
if(PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)
   if(APPLE AND CLANG_OMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libiomp5")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp=libiomp5")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
   else(APPLE AND CLANG_OMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
   endif(APPLE AND CLANG_OMP)
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

# Header files
include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})

# Link to binary files
link_directories(${PV_BINARY_DIR})


# Add executable
file(GLOB libSrcCPP ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB libSrcC ${PROJECT_SOURCE_DIR}/*.c)
add_executable(MLPLCA ${libSrcCPP} ${libSrcC})

# Link executable to PV 
target_link_libraries(MLPLCA pv)

# Link executable to external libraries
target_link_libraries(MLPLCA ${GDAL_LIBRARY})

