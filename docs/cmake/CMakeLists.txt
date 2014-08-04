# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)

# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)

# Set variable for current working directory (workspace directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/PetaVision CACHE FILEPATH "The base petavision directory")

set(OPEN_MP_THREADS FALSE CACHE PATH "Defines if PetaVision uses OpenMP threads")
set(LIB_SND_FILE FALSE CACHE PATH "Defines if PetaVision uses libSndFile for sound")

if(OPEN_MP_THREADS)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DPV_USE_THREADS")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -DPV_USE_THREADS")
endif (OPEN_MP_THREADS)

if (LIB_SND_FILE)
    MESSAGE("-- Searching for SND library ............")
    find_package(LIBSNDFILE)
    ADD_DEFINITIONS(-DPV_USE_SNDFILE)
    include_directories(${LIBSNDFILE_INCLUDE_DIR})
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
add_subdirectory(DepthLCA)
add_subdirectory(HyPerHLCA)
add_subdirectory(MaskLCA)
add_subdirectory(MLPLCA)
add_subdirectory(textAnalysis)
add_subdirectory(soundAnalysis)

add_subdirectory(PVSystemTests)
