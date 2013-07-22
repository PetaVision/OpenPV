## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)

set(PV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision/src")
set(PV_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision/lib")
set(BIDS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(BIDS_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Debug")

set(EXECUTABLE_OUTPUT_PATH ${BIDS_BINARY_DIR})

# Header files
include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})
include_directories(${MPI_C_INCLUDE_PATH})
include_directories(${MPI_CXX_INCLUDE_PATH})

# Link to binary files
link_directories(${PV_BINARY_DIR})


# Add executable
add_executable(BIDS BIDS_main_buildandrun.cpp)

# Link executable to PV 
target_link_libraries(BIDS PetaVision)

# Link executable to external libraries
target_link_libraries(BIDS ${GDAL_LIBRARY})
target_link_libraries(BIDS ${MPI_C_LIBRARIES})
target_link_libraries(BIDS ${MPI_CXX_LIBRARIES})
