## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)

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
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if (NOT GDAL_FOUND)
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)

if (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)
    find_package(MPI REQUIRED)
set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
endif (NOT MPI_C_FOUND OR NOT MPI_CXX_FOUND)

if (LIB_SND_FILE)
    find_package(LIBSNDFILE)
endif (LIB_SND_FILE)

include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})
IF(LIB_SND_FILE)
    include_directories(${LIBSNDFILE_INCLUDE_DIR})
ENDIF(LIB_SND_FILE)

# Link to binary files
link_directories(${PV_BINARY_DIR})


# Add executable
file(GLOB libSrcCPP ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB libSrcC ${PROJECT_SOURCE_DIR}/*.c)
add_executable(CloneVLayerTest ${libSrcCPP} ${libSrcC})

# Link executable to PV 
target_link_libraries(CloneVLayerTest pv)

# Link executable to external libraries
target_link_libraries(CloneVLayerTest ${GDAL_LIBRARY})
IF(LIB_SND_FILE)
   target_link_libraries(CloneVLayerTest ${LIBSNDFILE_LIBRARY})
ENDIF(LIB_SND_FILE)

