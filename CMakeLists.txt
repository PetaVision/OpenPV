## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.6)

## Version 0.1.0.0
set(PV_VERSION_MAJOR 0)
set(PV_VERSION_MINOR 1)
set(PV_VERSION_PATCH 0)
set(PV_VERSION_TWEAK 0)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel
endif (NOT CMAKE_BUILD_TYPE)

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

# File which denotes compile-ready folders
set(PV_SRC_LIST PV_FOLDER_LIST.txt)

set(PV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PV_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

######################################################################################
######################################################################################
##
## MAIN CODE
##
######################################################################################
######################################################################################

set(LIBRARY_OUTPUT_PATH ${PV_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PV_BINARY_DIR})

# Read in file and parse by line
file(READ "${PV_SRC_LIST}" contents)
STRING(REGEX REPLACE "\n" ";" contents "${contents}")

# Find PetaVision c/cpp files
foreach(LIN ${contents})
   file(GLOB_RECURSE libSrcCPP ${LIN}/*.cpp)
   file(GLOB_RECURSE libSrcC ${LIN}/*.c)
   set(PVLibSrc ${PVLibSrc} ${libSrcCPP})
   set(PVLibSrc ${PVLibSrc} ${libSrcC})
endforeach(LIN ${contents})

# Header file to pass CMake settings to source code
configure_file (
   "cMakeHeader.template"
   "${PV_SOURCE_DIR}/include/cMakeHeader.h"
)

# Add external libraries 
if (NOT GDAL_FOUND)
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)
include_directories(${GDAL_INCLUDE_DIR})

if (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
    find_package(MPI REQUIRED)
endif (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)

SET(LIBSnDFile_FIND_REQUIRED FALSE)
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

include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${MPI_C_INCLUDE_PATH})
IF(LIBSNDFIND_FOUND)
    include_directories(${LIBSNDFILE_INCLUDE_DIR})
ENDIF(LIBSNDFIND_FOUND)


# Add PetaVision
add_library(pv ${PVLibSrc})
