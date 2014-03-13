# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)

# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.6)

# Set variable for current working directory (workspace directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/PetaVision CACHE FILEPATH "The base petavision directory")

# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
#    

set(MPI_C_COMPILER_NAME "${CMAKE_C_COMPILER}")
set(MPI_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}")

add_subdirectory(${PV_DIR})
# # Insert additional add_subdirectory commands to add other projects.
add_subdirectory(BIDS)
add_subdirectory(DepthLCA)
add_subdirectory(HyPerHLCA)
add_subdirectory(MaskLCA)
add_subdirectory(textAnalysis)

add_subdirectory(PVSystemTests)
