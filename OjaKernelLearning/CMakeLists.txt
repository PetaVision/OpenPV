## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_FLAGS_DEBUG "-g3")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")

set(CMAKE_BUILD_TYPE ${BUILD_TYPE_OJAKERNELLEARNING}) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel

set(PV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision/src")
set(PV_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision/lib")
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Debug")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

# Header files
include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})
include_directories(${MPI_C_INCLUDE_PATH})
include_directories(${MPI_CXX_INCLUDE_PATH})

# Link to binary files
link_directories(${PV_BINARY_DIR})

# Add executable
add_executable(OjaKernelLearning ${PROJECT_SOURCE_DIR}/OjaKernelLearning.cpp)

# Link executable to PV 
target_link_libraries(OjaKernelLearning pv)

# Link executable to external libraries
target_link_libraries(OjaKernelLearning ${GDAL_LIBRARY})
target_link_libraries(OjaKernelLearning ${MPI_C_LIBRARIES})
target_link_libraries(OjaKernelLearning ${MPI_CXX_LIBRARIES})

