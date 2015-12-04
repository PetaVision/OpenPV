## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)

string(FIND ${CMAKE_CURRENT_SOURCE_DIR} "/" pos REVERSE)
MATH(EXPR pos ${pos}+1)
string(SUBSTRING ${CMAKE_CURRENT_SOURCE_DIR} ${pos} -1 TEST_TARGET_NAME)

set(BUILD_DIR "" CACHE STRING "The directory to write the executable binary into. (if blank, CMAKE_BUILD_TYPE will give the build directory")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel
endif (NOT CMAKE_BUILD_TYPE)

set(PV_SOURCE_DIR "${PV_DIR}/src")
set(PV_BINARY_DIR "${PV_DIR}/lib")
set(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Define the directory for the executable file.
# If BUILD_DIR is empty, use CMAKE_BUILD_TYPE string as the name.
# If BUILD_DIR starts with "/", use it as an absolute path
# If BUILD_DIR is nonempty and doesn't start with "/", use it as a path relative to the current directory
if("${BUILD_DIR}" STREQUAL "")
   set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
else("${BUILD_DIR}" STREQUAL "")
   string(SUBSTRING "${BUILD_DIR}" 0 1 BUILD_DIR_FIRSTCHAR)
   if("${BUILD_DIR_FIRSTCHAR}" STREQUAL "/")
      set(PROJECT_BINARY_DIR "${BUILD_DIR}")
   else("${BUILD_DIR_FIRSTCHAR}" STREQUAL "/")
      set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${BUILD_DIR}")
   endif("${BUILD_DIR_FIRSTCHAR}" STREQUAL "/")
endif("${BUILD_DIR}" STREQUAL "")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

include_directories(${PV_SOURCE_DIR})
include_directories(${GDAL_INCLUDE_DIR})

# Link to binary files
link_directories(${PV_BINARY_DIR})

# Add executable
file(GLOB libSrcCPP ${TEST_SOURCE_DIR}/*.cpp)
file(GLOB libSrcC ${TEST_SOURCE_DIR}/*.c)

if(PV_USE_CUDA)
   cuda_add_executable(${TEST_TARGET_NAME} ${libSrcCPP} ${libSrcC})
else(PV_USE_CUDA)
   add_executable(${TEST_TARGET__NAME} ${libSrcCPP} ${libSrcC})
endif(PV_USE_CUDA)

target_link_libraries(${TEST_TARGET_NAME} pv)

# still needed on Mac if pv is built as a shared object library
target_link_libraries(${TEST_TARGET_NAME} ${GDAL_LIBRARY})

## target_link_libraries command no longer needed because the libraries are included in libpv.so
#IF(PV_USE_CUDNN)
#   target_link_libraries(${PV_PROJECT_NAME} ${CUDNN_PATH}/libcudnn.so)
#ENDIF(PV_USE_CUDNN)
#
