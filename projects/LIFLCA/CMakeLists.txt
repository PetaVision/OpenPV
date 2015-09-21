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
    set(CMAKE_BUILD_TYPE Debug)
endif (NOT CMAKE_BUILD_TYPE)

set(PV_SOURCE_DIR "${PV_DIR}/src")
set(PV_BINARY_DIR "${PV_DIR}/lib")
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src") # should we move source files to a src directory?
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if (NOT GDAL_FOUND)
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)

if (PV_USE_MPI)
    if (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
        find_package(MPI REQUIRED)
        set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
        set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
    endif (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
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

# Link executable to external libraries
target_link_libraries(${PV_PROJECT_NAME} ${GDAL_LIBRARY})

IF(PV_USE_CUDNN)
   IF(APPLE)
      file(GLOB_RECURSE CUDNN_LIBRARY "${CUDNN_PATH}*/libcudnn.dylib")
   ELSE(APPLE)
      file(GLOB_RECURSE CUDNN_LIBRARY "${CUDNN_PATH}*/libcudnn.so")
   ENDIF(APPLE)
   target_link_libraries(${PV_PROJECT_NAME} ${CUDNN_LIBRARY})
ENDIF(PV_USE_CUDNN)
