## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)

## Version 0.1.0.0
set(PV_VERSION_MAJOR 0)
set(PV_VERSION_MINOR 1)
set(PV_VERSION_PATCH 0)
set(PV_VERSION_TWEAK 0)

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/docs/cmake/Modules/")

if (NOT CMAKE_BUILD_TYPE)        #To use a different CMAKE_BUILD_TYPE, use the option -DCMAKE_BUILD_TYPE=xxxx on the command line
   set(CMAKE_BUILD_TYPE Release) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel
endif (NOT CMAKE_BUILD_TYPE)

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

# File which denotes compile-ready folders
set(PV_SRC_LIST PV_FOLDER_LIST.txt)

set(PV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PV_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")

######################################################################################
######################################################################################
##
## MAIN CODE
##
######################################################################################
######################################################################################

set(LIBRARY_OUTPUT_PATH ${PV_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PV_BINARY_DIR})

# Lex/Yacc for parser
if (NOT BISON_PARSER_DEFINED)
    MESSAGE("-- Searching for bison ..................")
    find_package(BISON REQUIRED)
    BISON_TARGET(PARSER ${PV_SOURCE_DIR}/io/parser/params.y ${PV_SOURCE_DIR}/io/parser-generated/param_parser.cpp COMPILE_FLAGS "-y")

endif (NOT BISON_PARSER_DEFINED)
if (NOT FLEX_SCANNER_DEFINED)
    MESSAGE("-- Searching for flex (yacc) ............")
    find_package(FLEX REQUIRED)
    FLEX_TARGET(SCANNER ${PV_SOURCE_DIR}/io/parser/params.l ${PV_SOURCE_DIR}/io/parser-generated/param_lexer.c)
    ADD_FLEX_BISON_DEPENDENCY(SCANNER PARSER)
endif (NOT FLEX_SCANNER_DEFINED)


# Read in file and parse by line
file(READ "${PV_SRC_LIST}" contents)
STRING(REGEX REPLACE "\n" ";" contents "${contents}")

# Find PetaVision c/cpp files
foreach(LIN ${contents})
   file(GLOB libSrcCPP ${LIN}/*.cpp)
   file(GLOB libSrcC ${LIN}/*.c)
   set(PVLibSrc ${PVLibSrc} ${libSrcCPP})
   set(PVLibSrc ${PVLibSrc} ${libSrcC})

   if(PV_USE_CUDA)
      file(GLOB_RECURSE libSrcCU ${LIN}/*.cu)
      set(PVLibSrcCu ${PVLibSrcCu} ${libSrcCU})
   endif(PV_USE_CUDA)
endforeach(LIN ${contents})

find_package(Git)
if (GIT_FOUND)
   unset(PV_GIT_REVISION)
   # Get current commit
   execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                   RESULT_VARIABLE PV_CURRENT_COMMIT_RESULT
                   OUTPUT_VARIABLE PV_CURRENT_COMMIT
                   ERROR_VARIABLE PV_CURRENT_COMMIT_ERROR
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
   if (${PV_CURRENT_COMMIT_RESULT} EQUAL 0)
      # Get commit hash and date commit was authored
      execute_process(COMMAND "${GIT_EXECUTABLE}" log -n 1 "--format=%H (%ad)" "${PV_CURRENT_COMMIT}"
                      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      RESULT_VARIABLE PV_GIT_REVISION_RESULT
                      OUTPUT_VARIABLE PV_GIT_REVISION
                      ERROR_VARIABLE PV_GIT_REVISION_ERROR
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(PV_GIT_REVISION "git repository version ${PV_GIT_REVISION}")
      # See if there are any local changes
      execute_process(COMMAND "${GIT_EXECUTABLE}" status -s
                      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      RESULT_VARIABLE PV_GIT_STATUS_RESULT
                      OUTPUT_VARIABLE PV_GIT_STATUS_OUTPUT
                      ERROR_VARIABLE VP_GIT_STATUS_ERROR
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(LENGTH "${PV_GIT_STATUS_OUTPUT}" PV_STATUS_LENGTH)
      if(NOT ("${PV_STATUS_LENGTH}" EQUAL 0))
         set(PV_GIT_REVISION "${PV_GIT_REVISION} with local modifications")
      endif(NOT ("${PV_STATUS_LENGTH}" EQUAL 0))
   else (${PV_CURRENT_COMMIT_RESULT} EQUAL 0)
      unset(PV_GIT_REVISION)
   endif (${PV_CURRENT_COMMIT_RESULT} EQUAL 0)
endif (GIT_FOUND)

find_package(Subversion)
if (Subversion_FOUND)
   unset(PV_SVN_REVISION)
   set(SVNVERSION_COMMAND "${Subversion_SVN_EXECUTABLE}version")
   # !! lack of space between ${Subversion_SVN_EXECUTABLE} and version is deliberate:
   # !! Subversion_SVN_EXECUTABLE returns svn, and we want to call svnversion.
   execute_process(COMMAND "${SVNVERSION_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}"
                   OUTPUT_VARIABLE PV_SVN_REVISION
                   ERROR_VARIABLE PV_SVN_ERROR
                   RESULT_VARIABLE PV_SVN_RESULT
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
   if (${PV_SVN_RESULT} EQUAL 0)
      if ("${PV_SVN_REVISION}" STREQUAL "Unversioned directory")
         unset(PV_SVN_REVISION)
      else ("${PV_SVN_REVISION}" STREQUAL "Unversioned directory")
         string(REGEX REPLACE "M$" " with local modifications" PV_SVN_REVISION "${PV_SVN_REVISION}")
         set(PV_SVN_REVISION "svn repository version ${PV_SVN_REVISION}")
      endif ("${PV_SVN_REVISION}" STREQUAL "Unversioned directory")
   else (${PV_SVN_RESULT} EQUAL 0)
      unset(PV_SVN_REVISION)
   endif (${PV_SVN_RESULT} EQUAL 0)
else (Subversion_FOUND)
   unset(PV_SVN_REVISION)
endif (Subversion_FOUND)

if (DEFINED PV_GIT_REVISION)
   set(PV_REVISION "${PV_GIT_REVISION}")
endif (DEFINED PV_GIT_REVISION)
if (DEFINED PV_SVN_REVISION)
   set(PV_REVISION "${PV_SVN_REVISION}")
endif (DEFINED PV_SVN_REVISION)
if (DEFINED PV_SVN_REVISION AND DEFINED PV_GIT_REVISION) # This probably shouldn't happen
   set(PV_REVISION "${PV_GIT_REVISION}/${PV_SVN_REVISION}")
endif (DEFINED PV_SVN_REVISION AND DEFINED PV_GIT_REVISION)
if (NOT (DEFINED PV_SVN_REVISION OR DEFINED PV_GIT_REVISION))
   set(PV_REVISION "unknown version")
endif (NOT (DEFINED PV_SVN_REVISION OR DEFINED PV_GIT_REVISION))

# Header file to pass CMake settings to source code
configure_file (
   "cMakeHeader.template"
   "${PV_SOURCE_DIR}/include/cMakeHeader.h"
)

# START DELETE ME
if(FALSE)
# Handled in top level CMakeLists.txt
# Add external libraries 
if (NOT GDAL_FOUND)
    MESSAGE("-- Searching for GDAL library ...........")
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)
include_directories(${GDAL_INCLUDE_DIR})
endif(FALSE)
# END DELETE ME

# 
# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
#

# START MAYBE DELETE / MOVE TO TOPLEVEL
if (PV_USE_MPI)
    if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
        MESSAGE("-- Searching for MPI library ............")
        find_package(MPI REQUIRED)
    endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)

    # Can this be moved to the top level CMakeListst.txt? If so,
    # then this entire if block can be removed
    set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
    set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
endif (PV_USE_MPI)
# END MAYBE DELETE

# START MAYBE MOVE TO TOPLEVEL
# Could this find package be moved to the top level so that 
# this does not need to be repeated for parallel directories
# that need CUDA?
if (PV_USE_CUDA)
    MESSAGE("-- Searching for CUDA library ............")
    find_package(CUDA)
endif (PV_USE_CUDA)

#IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
#   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g3; -G3; -O0)
#ELSE(CMAKE_BUILD_TYPE MATCHES RELEASE)
#   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g0; -G0; -O3)
#endif(CMAKE_BUILD_TYPE)

if(PV_USE_CUDA)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_30; )
else(PV_USE_CUDA)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_20;)
endif(PV_USE_CUDA)
# END MAYBE MOVE TO TOPLEVEL

# START MAYBE MOVE TO TOPLEVEL
if(CUDA_RELEASE)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O;)
else(CUDA_RELEASE)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xptxas; -v; -keep; -lineinfo; -g; -G;)
endif(CUDA_RELEASE)
#cuda_compile(CUDA_O ${PVLibSrcCu}) 
# END MAYBE MOVE TO TOPLEVEL

# Add PetaVision
if(PV_USE_CUDA)
   if(PV_BUILD_SHARED)
      cuda_add_library(pv SHARED ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS} ${PVLibSrcCu})
   else(PV_BUILD_SHARED)
      cuda_add_library(pv STATIC ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS} ${PVLibSrcCu})
   endif(PV_BUILD_SHARED)
else(PV_USE_CUDA)
   if(PV_BUILD_SHARED)
      add_library(pv SHARED ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS})
   else(PV_BUILD_SHARED)
      add_library(pv STATIC ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS})
   endif(PV_BUILD_SHARED)
endif(PV_USE_CUDA)

# Link executable to external libraries
target_link_libraries(pv ${GDAL_LIBRARY})

# START MAYBE DELETE
# Could this be moved to the top level so that parallel directories
# that need cudnn don't have to repeat this 
IF(PV_USE_CUDA)
   IF(APPLE)
      set(CUDNN_LIB_NAME "libcudnn.dylib")
   ELSE(APPLE)
      set(CUDNN_LIB_NAME "libcudnn.so")
   ENDIF(APPLE)

   set(CUDNN_LIBRARY_SEARCH_PATH 
      ${CUDNN_PATH}
      ${CUDNN_PATH}/lib
      ${CUDNN_PATH}/lib64
   )

   find_library(CUDNN_LIBRARY ${CUDNN_LIB_NAME} ${CUDNN_LIBRARY_SEARCH_PATH})

   target_link_libraries(pv ${CUDNN_LIBRARY})
ENDIF(PV_USE_CUDA)
# END MAYBE DELETE

