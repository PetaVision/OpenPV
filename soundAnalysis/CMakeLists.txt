## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)

IF(NOT LIBSNDFILE_FOUND)
    MESSAGE(FATAL_ERROR "Sndfile library required for soundAnalysis sandbox")
ENDIF(NOT LIBSNDFILE_FOUND)

set(PV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision/src")
set(PV_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision/lib")
set(TA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(TA_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Debug")

set(EXECUTABLE_OUTPUT_PATH ${TA_BINARY_DIR})

# Header files
include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})
include_directories(${MPI_C_INCLUDE_PATH})
include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${LIBSNDFILE_INCLUDE_DIR})

# Link to PV binary files
link_directories(${PV_BINARY_DIR})

# Add executable
add_executable(soundAnalysis main.cpp)

# Link executable to PV 
target_link_libraries(soundAnalysis pv)

# Link executable to external libraries
target_link_libraries(soundAnalysis ${GDAL_LIBRARY})
target_link_libraries(soundAnalysis ${MPI_C_LIBRARIES})
target_link_libraries(soundAnalysis ${MPI_CXX_LIBRARIES})
target_link_libraries(soundAnalysis ${LIBSNDFILE_LIBRARY})
