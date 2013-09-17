## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.6)

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PV_WRKSPC_DIR}/PetaVision/docs/cmake/Modules/")

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif (NOT CMAKE_BUILD_TYPE)

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

if (NOT LIBSNDFILE_FOUND)
    find_package(LIBSNDFILE)
endif (NOT LIBSNDFILE_FOUND)

include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${MPI_C_INCLUDE_PATH})
IF(LIBSNDFILE_FOUND)
    include_directories(${LIBSNDFILE_INCLUDE_DIR})
ENDIF(LIBSNDFILE_FOUND)

# Header files
include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})
#include_directories(${MPI_C_INCLUDE_PATH})
#include_directories(${MPI_CXX_INCLUDE_PATH})

# Link to binary files
link_directories(${PV_BINARY_DIR})


# Add executable
file(GLOB libSrcCPP ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB libSrcC ${PROJECT_SOURCE_DIR}/*.c)
add_executable(NormalizeSystemTest ${libSrcCPP} ${libSrcC})

# Link executable to PV 
target_link_libraries(NormalizeSystemTest pv)

# Link executable to external libraries
target_link_libraries(NormalizeSystemTest ${GDAL_LIBRARY})
target_link_libraries(NormalizeSystemTest ${MPI_C_LIBRARIES})
target_link_libraries(NormalizeSystemTest ${MPI_CXX_LIBRARIES})
IF(LIBSNDFILE_FOUND)
   target_link_libraries(NormalizeSystemTest ${LIBSNDFILE_LIBRARY})
ENDIF(LIBSNDFILE_FOUND)

