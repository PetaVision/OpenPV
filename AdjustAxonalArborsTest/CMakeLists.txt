## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.6)

#Set default PV_WRKSPC_DIR if not already set
if (NOT PV_WRKSPC_DIR)
    set(PV_WRKSPC_DIR "..")
    message("*** Setting PV_WRKSPC_DIR to default value \"${PV_WRKSPC_DIR}\".")
    message("*** PV_WRKSPC_DIR must be the directory containing the PetaVision directory.")
    message("*** Set PV_WRKSPC_DIR manually if PetaVision is in a different location.")
endif (NOT PV_WRKSPC_DIR)

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PV_WRKSPC_DIR}/PetaVision/docs/cmake/Modules/")

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif (NOT CMAKE_BUILD_TYPE)

set(PV_SOURCE_DIR "${PV_WRKSPC_DIR}/PetaVision/src")
set(PV_BINARY_DIR "${PV_WRKSPC_DIR}/PetaVision/lib")
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
add_executable(AdjustAxonalArborsTest ${libSrcCPP} ${libSrcC})

# Link executable to PV 
target_link_libraries(AdjustAxonalArborsTest pv)

# Link executable to external libraries
target_link_libraries(AdjustAxonalArborsTest ${GDAL_LIBRARY})
target_link_libraries(AdjustAxonalArborsTest ${MPI_C_LIBRARIES})
target_link_libraries(AdjustAxonalArborsTest ${MPI_CXX_LIBRARIES})
IF(LIBSNDFILE_FOUND)
   target_link_libraries(AdjustAxonalArborsTest ${LIBSNDFILE_LIBRARY})
ENDIF(LIBSNDFILE_FOUND)

