# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)


# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)


# Set variable for PetaVision directory (the directory that contains
# the core library's source directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pv-core CACHE PATH "The core PetaVision directory")

# Platform dependent values
#  OpenMP compiler flag is platform dependent
#  Default cudnn path is platform dependent

if(APPLE)
   set(PV_OPENMP_COMPILER_FLAG_DEFAULT "-fopenmp=libiomp5")
   set(CUDNN_PATH_DEFAULT "/usr/local/cudnn-7.0/")
 else(APPLE)
   set(PV_OPENMP_COMPILER_FLAG_DEFAULT "-fopenmp")
   set(CUDNN_PATH_DEFAULT "/nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/")
endif(APPLE)

#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Note: It would be preferable to parse ${PV_DIR}/src/include/cMakeHeader.h for these values.
set(PV_USE_MPI TRUE CACHE BOOL "Defines whether PetaVision uses MPI")
set(PV_USE_OPENMP_THREADS TRUE CACHE BOOL "Defines if PetaVision uses OpenMP threads")
set(PV_USE_OPENCL FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
set(PV_USE_CUDA FALSE CACHE BOOL "Defines if PetaVision uses CUDA GPU")
set(CUDA_RELEASE FALSE CACHE BOOL "Defines if Cuda compiles with optimization")
set(PV_OPENMP_COMPILER_FLAG ${PV_OPENMP_COMPILER_FLAG_DEFAULT} CACHE STRING "Compiler flag for compiling with OpenMP")

#set(PV_USE_CUDNN FALSE CACHE BOOL "Defines if PetaVision uses Cuda's CUDNN library")
set(CUDNN_PATH ${CUDNN_PATH_DEFAULT} CACHE PATH "Location of cuDNN libraries")

set(CLANG_OMP TRUE CACHE BOOL "Defines if we're compiling with Clang + omp support")

set(OpenCL_dir /usr/local/cuda-6.5/ CACHE PATH "Location of opencl headers")

if(PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)
   if(APPLE AND CLANG_OMP)
      message(WARNING "Using OpenMP threads with Clang. Make sure that your version of Clang supports OpenMP.")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -stdlib=libstdc++")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")
   endif(APPLE AND CLANG_OMP)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PV_OPENMP_COMPILER_FLAG}")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PV_OPENMP_COMPILER_FLAG}")
   set(PV_USE_OPENMP_THREADS_FLAGS_ADDED TRUE)
endif (PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)

set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel
endif (NOT CMAKE_BUILD_TYPE)


if(PV_USE_OPENCL AND NOT DEFINED PV_USE_OPENCL_FLAGS_ADDED)
   IF(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework opencl")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -framework opencl")
   ELSE(APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${OpenCL_dir}/include -L/${OpenCL_dir}/lib64 -lOpenCL")
   ENDIF(APPLE)
   set(PV_USE_OPENCL_FLAGS_ADDED TRUE)
endif (PV_USE_OPENCL AND NOT DEFINED PV_USE_OPENCL_FLAGS_ADDED)

# If CUDA is being used, the PV_USE_CUDNN needs to be set
if(PV_USE_CUDA)
   set(PV_USE_CUDNN TRUE)
endif(PV_USE_CUDA)

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

# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc DCMAKE_CXX_COMPILER=openmpic++
#    
if (PV_USE_MPI)
    if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
        MESSAGE("-- Searching for MPI library ............")
        find_package(MPI REQUIRED)
    endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)

   # The user may specify a non-standard compiler name for MPI
   #  For example, the user may type:
   #    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc -DCMAKE_CXX_COMPILER=openmpic++
   #

   set(MPI_C_COMPILER_NAME "${CMAKE_C_COMPILER}")
   set(MPI_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}")

   if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
       find_package(MPI REQUIRED)
   endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
   set(CMAKE_C_COMPILER "${MPI_C_COMPILER}")
   set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

endif (PV_USE_MPI)

if (NOT GDAL_FOUND)
    MESSAGE("-- Searching for GDAL library ...........")
    find_package(GDAL REQUIRED)
endif (NOT GDAL_FOUND)
include_directories(${GDAL_INCLUDE_DIR})

# Adds the core PetaVision directory
add_subdirectory(${PV_DIR})


# Adds other projects that link to PetaVision.  Their CMakeLists.txt
# file should use the cache variable PV_DIR to point to PetaVision.
# Create a text file called subdirectories.txt in the same directory
# as this CMakeLists.txt file.
# 
# Comments are marked with the number sign '#'; it and everything
# following it are ignored.  Leading and trailing whitespace is then
# stripped  If the resulting line is nonempty, its contents are interpreted
# as a directory and and passed to the CMake add_subdirectory command.
set(SUBDIR_TXT "${CMAKE_CURRENT_BINARY_DIR}/subdirectories.txt")
if (EXISTS ${SUBDIR_TXT})
    set(PV_SUBDIR_LIST ${SUBDIR_TXT})
    file(READ "${PV_SUBDIR_LIST}" contents)
    string(REGEX REPLACE "\n" ";" contents "${contents}")
    
    foreach(LIN ${contents})
        string(FIND "${LIN}" "#" NUMBERSIGNLOC)
        if (NOT (${NUMBERSIGNLOC} LESS 0) )
            string(SUBSTRING "${LIN}" 0 ${NUMBERSIGNLOC} LIN)
        endif (NOT (${NUMBERSIGNLOC} LESS 0) )
        string(STRIP "${LIN}" LIN)
        if (NOT ("${LIN}" STREQUAL "") )
            message("Adding subdirectory \"${LIN}\"")
            add_subdirectory(${LIN} ${LIN})
        endif (NOT ("${LIN}" STREQUAL "") )
    endforeach(LIN ${contents})
    
    # Create a text file called ${SUBDIR_TXT} in the same directory as
    # this CMakeLists.txt file.  That file should contain a list of
    # the projects to include.
    # ${SUBDIR_TXT} is in the .gitignore file so that it won't be
    # (and shouldn't be) committed to the repository.
    # The sample-${SUBDIR_TXT} file illustrates the format.
    # Note that every time you modify ${SUBDIR_TXT} you will need to
    # rerun cmake.
else (EXISTS ${SUBDIR_TXT})
   message("Only building the core PV library.  Create a ${SUBDIR_TXT} file to build additional projects linking to it.")
endif (EXISTS ${SUBDIR_TXT})
