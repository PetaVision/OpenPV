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


# Variables to set on the command line
set(PV_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "The base PetaVision directory")
set(PV_USE_MPI TRUE CACHE BOOL "Defines whether PetaVision uses MPI")
set(PV_USE_OPENMP_THREADS FALSE CACHE BOOL "Defines whether PetaVision uses OpenMP threads")
set(PV_USE_OPENCL FALSE CACHE BOOL "Defines whether PetaVision uses OpenCL GPU")
set(OpenCL_dir /usr/local/cuda-6.5/ CACHE PATH "Location of OpenCL headers")
set(PV_USE_CUDA FALSE CACHE BOOL "Defines whether PetaVision uses CUDA GPU")
set(CUDA_RELEASE FALSE CACHE BOOL "Defines whether CUDA compiles with optimization")
set(PV_USE_CUDNN FALSE CACHE BOOL "Defines whether PetaVision uses CUDA's cuDNN library")
set(CUDNN_PATH /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/ CACHE PATH "Location of cuDNN libraries")
set(PV_BUILD_SHARED FALSE CACHE BOOL "Defines whether the pv object is built as a shared object")
set(CLANG_OMP TRUE CACHE BOOL "Defines whether we're compiling with Clang + omp support")

# Sound-related classes moved to auxlib/pvsound Mar 16, 2015 
# set(LIB_SND_FILE FALSE CACHE BOOL "Defines if PetaVision uses LIB_SND_FILE")

if(PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)
   if(APPLE AND CLANG_OMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libiomp5")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp=libiomp5")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
   else(APPLE AND CLANG_OMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
   endif(APPLE AND CLANG_OMP)
   set(PV_USE_OPENMP_THREADS_FLAGS_ADDED TRUE)
endif (PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)

if(PV_USE_OPENCL AND NOT DEFINED PV_USE_OPENCL_FLAGS_ADDED)
   IF(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework opencl")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -framework opencl")
   ELSE(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL")
   ENDIF(APPLE)
   set(PV_USE_OPENCL_FLAGS_ADDED)
endif (PV_USE_OPENCL AND NOT DEFINED PV_USE_OPENCL_FLAGS_ADDED)


set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

if(PV_USE_CUDA)
    #Set include file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE}")
    if(PV_USE_CUDNN)
       file(GLOB_RECURSE CUDNN_HEADER "${CUDNN_PATH}*/cudnn.h")
       string(REGEX REPLACE "/[^/]+$" "" CUDNN_INCLUDE_PATH "${CUDNN_HEADER}")
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDNN_INCLUDE_PATH}")
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDNN_INCLUDE_PATH}")
    endif(PV_USE_CUDNN)
endif(PV_USE_CUDA)

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
   set(PV_REVISION "${PV_GIT_REVISION}/${PV_SVN_REVISION")
endif (DEFINED PV_SVN_REVISION AND DEFINED PV_GIT_REVISION)
if (NOT (DEFINED PV_SVN_REVISION OR DEFINED PV_GIT_REVISION))
   set(PV_REVISION "unknown version")
endif (NOT (DEFINED PV_SVN_REVISION OR DEFINED PV_GIT_REVISION))

# Header file to pass CMake settings to source code
configure_file (
   "cMakeHeader.template"
   "${PV_SOURCE_DIR}/include/cMakeHeader.h"
)

# Add external libraries 
if (NOT GDAL_FOUND)
    MESSAGE("-- Searching for GDAL library ...........")
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)
include_directories(${GDAL_INCLUDE_DIR})

# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
#

if (PV_USE_MPI)
    if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
        MESSAGE("-- Searching for MPI library ............")
        find_package(MPI REQUIRED)
    endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)

    set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
    set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
endif (PV_USE_MPI)

# Sound-related classes moved to auxlib/pvsound Mar 16, 2015 
# if (LIB_SND_FILE AND NOT DEFINED LIBSNDFILE_FOUND)
#     MESSAGE("-- Searching for SND library ............")
#     find_package(LIBSNDFILE)
# endif (LIB_SND_FILE AND NOT DEFINED LIBSNDFILE_FOUND)

if (PV_USE_CUDA)
    MESSAGE("-- Searching for CUDA library ............")
    find_package(CUDA)
endif (PV_USE_CUDA)

# Sound-related classes moved to auxlib/pvsound Mar 16, 2015 
# #Include compiler directive and include directory for LIBSNDFILE
# if(LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)
#    include_directories(${LIBSNDFILE_INCLUDE_DIR})
#    add_definitions(-DPV_USE_SNDFILE)
#    set(LIB_SND_FILE_FLAGS_ADDED)
# endif (LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)

#IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
#   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g3; -G3; -O0)
#ELSE(CMAKE_BUILD_TYPE MATCHES RELEASE)
#   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g0; -G0; -O3)
#endif(CMAKE_BUILD_TYPE)

if(PV_USE_CUDNN)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_30; )
else(PV_USE_CUDNN)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_20;)
endif(PV_USE_CUDNN)


if(CUDA_RELEASE)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O;)
else(CUDA_RELEASE)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xptxas; -v; -keep; -lineinfo; -g; -G;)
endif(CUDA_RELEASE)
#cuda_compile(CUDA_O ${PVLibSrcCu}) 

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
# Sound-related classes moved to auxlib/pvsound Mar 16, 2015 
# IF(LIBSNDFILE_FOUND)
#    target_link_libraries(pv ${LIBSNDFILE_LIBRARY})
# ENDIF(LIBSNDFILE_FOUND)

IF(PV_USE_CUDNN)
   IF(APPLE)
      file(GLOB_RECURSE CUDNN_LIBRARY "${CUDNN_PATH}*/libcudnn.dylib")
   ELSE(APPLE)
      file(GLOB_RECURSE CUDNN_LIBRARY "${CUDNN_PATH}*/libcudnn.so")
   ENDIF(APPLE)
   target_link_libraries(pv ${CUDNN_LIBRARY})
ENDIF(PV_USE_CUDNN)

