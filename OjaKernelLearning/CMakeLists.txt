## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.6)


set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

# To allow for a single CMakeLists.txt to add_subdirectory several projects, where build type might not all be the same,
# can set BUILD_TYPE_OJAKERNELLEARNING separately and it will be used as set
if (BUILD_TYPE_OJAKERNELLEARNING)
    set(BUILD_TYPE_SAVE_OLD ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE ${BUILD_TYPE_OJAKERNELLEARNING}) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel
endif (BUILD_TYPE_OJAKERNELLEARNING)

set(PV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision/src")
set(PV_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../PetaVision/lib")
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if (NOT GDAL_FOUND)
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)

if (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
    find_package(MPI REQUIRED)
endif (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${MPI_C_INCLUDE_PATH})

SET(LIBSndFile_FIND_REQUIRED FALSE)
IF (NOT LIBSNDFILE_FOUND)
    # Base Io build system
    # Written by Jeremy Tregunna <jeremy.tregunna@me.com>
    #
    # Find libsndfile.
    FIND_PATH(LIBSNDFILE_INCLUDE_DIR sndfile.h)

    SET(LIBSNDFILE_NAMES ${LIBSNDFILE_NAMES} sndfile libsndfile)
    FIND_LIBRARY(LIBSNDFILE_LIBRARY NAMES ${LIBSNDFILE_NAMES} PATH)

    IF(LIBSNDFILE_INCLUDE_DIR AND LIBSNDFILE_LIBRARY)
        SET(LIBSNDFILE_FOUND TRUE CACHE INTERNAL "")
    ENDIF(LIBSNDFILE_INCLUDE_DIR AND LIBSNDFILE_LIBRARY)

    IF(LIBSNDFILE_FOUND)
        MESSAGE(STATUS "Found LibSndFile: ${LIBSNDFILE_LIBRARY}")
        #Add compiler definition PV_USE_SNDFILE
        ADD_DEFINITIONS(-DPV_USE_SNDFILE)
    ELSE(LIBSNDFILE_FOUND)
        MESSAGE(STATUS "Skipping Sndfile library")
        REMOVE_DEFINITIONS(-DPV_USE_SNDFILE)
    ENDIF (LIBSNDFILE_FOUND)
ENDIF (NOT LIBSNDFILE_FOUND)

# Header files
include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})
include_directories(${MPI_C_INCLUDE_PATH})
include_directories(${MPI_CXX_INCLUDE_PATH})
IF(LIBSNDFILE_FOUND)
    include_directories(${LIBSNDFILE_INCLUDE_DIR})
ENDIF(LIBSNDFILE_FOUND)

# Link to binary files
link_directories(${PV_BINARY_DIR})

# Add executable
file(GLOB libSrcCPP ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB libSrcC ${PROJECT_SOURCE_DIR}/*.c)
add_executable(OjaKernelLearning ${libSrcCPP} ${libSrcC})

# Link executable to PV 
target_link_libraries(OjaKernelLearning pv)

# Link executable to external libraries
target_link_libraries(OjaKernelLearning ${GDAL_LIBRARY})
target_link_libraries(OjaKernelLearning ${MPI_C_LIBRARIES})
target_link_libraries(OjaKernelLearning ${MPI_CXX_LIBRARIES})
IF(LIBSNDFILE_FOUND)
    target_link_libraries(OjaKernelLearning ${LIBSNDFILE_LIBRARY})
ENDIF(LIBSNDFILE_FOUND)

if (BUILD_TYPE_OJAKERNELLEARNING)
    set(CMAKE_BUILD_TYPE ${BUILD_TYPE_OJAKERNELLEARNING})
endif (BUILD_TYPE_OJAKERNELLEARNING)
