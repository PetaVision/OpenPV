# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)


# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)


# Set variable for PetaVision directory (the directory that contains
# the core library's source directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pv-core CACHE PATH "The core PetaVision directory")


#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${PV_DIR}/docs/cmake/Modules/")

set(PV_USE_OPENMP_THREADS TRUE CACHE BOOL "Defines if PetaVision uses OpenMP threads")
set(PV_USE_OPENCL FALSE CACHE BOOL "Defines if PetaVision uses OpenCL GPU")
set(PV_USE_CUDA FALSE CACHE BOOL "Defines if PetaVision uses CUDA GPU")
# Sound-related classes moved to auxlib/pvsound Mar 16, 2015
# set(LIB_SND_FILE FALSE CACHE BOOL "Defines if PetaVision uses LIB_SND_FILE")
set(CUDA_RELEASE FALSE CACHE BOOL "Defines if Cuda compiles with
optimization")

set(PV_USE_CUDNN FALSE CACHE BOOL "Defines if PetaVision uses Cuda's CUDNN library")
set(CUDNN_PATH /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/ CACHE PATH "Location of cuDNN libraries")

set(CLANG_OMP TRUE CACHE BOOL "Defines if we're compiling with Clang + omp support")

set(OpenCL_dir /usr/local/cuda-6.5/ CACHE PATH "Location of opencl headers")

if(PV_USE_OPENMP_THREADS AND NOT DEFINED PV_USE_OPENMP_THREADS_FLAGS_ADDED)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
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
   set(PV_USE_OPENCL_FLAGS_ADDED TRUE)
endif (PV_USE_OPENCL AND NOT DEFINED PV_USE_OPENCL_FLAGS_ADDED)

if(PV_USE_CUDA)
    #Set include file
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDA_TOOLKIT_INCLUDE}")
    if(PV_USE_CUDNN)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CUDNN_PATH}")
       set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I ${CUDNN_PATH}")
    endif(PV_USE_CUDNN)
endif(PV_USE_CUDA)

# Sound-related classes moved to auxlib/pvsound Mar 16, 2015 
# if (LIB_SND_FILE AND NOT LIBSNDFILE_FOUND)
#     find_package(LIBSNDFILE)
# endif (LIB_SND_FILE AND NOT LIBSNDFILE_FOUND)
# 
# if(LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)
#    include_directories(${LIBSNDFILE_INCLUDE_DIR})
#    add_definitions(-DPV_USE_SNDFILE)
#    set(LIB_SND_FILE_FLAGS_ADDED TRUE)
# endif (LIB_SND_FILE AND LIBSNDFILE_FOUND AND NOT DEFINED LIB_SND_FILE_FLAGS_ADDED)

# The user may specify a non-standard compiler name for MPI
#  For example, the user may type:
#    cmake CMakeLists.txt -DCMAKE_C_COMPILER=openmpicc DCMAKE_CXX_COMPILER=openmpic++
#    

if (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)
    MESSAGE("-- Searching for MPI library ............")
    find_package(MPI REQUIRED)
endif (NOT DEFINED MPI_C_COMPILER OR NOT DEFINED MPI_CXX_COMPILER)


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
if (EXISTS subdirectories.txt)
    set(PV_SUBDIR_LIST subdirectories.txt)
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
    
    # Create a text file called subdirectories.txt in the same directory as
    # this CMakeLists.txt file.  That file should contain a list of
    # the projects to include.
    # subdirectories.txt is in the .gitignore file so that it won't be
    # (and shouldn't be) committed to the repository.
    # The sample-subdirectories.txt file illustrates the format.
    # Note that every time you modify subdirectories.txt you will need to
    # rerun cmake.
else (EXISTS subdirectories.txt)
   message("Only building the core PV library.  Create a subdirectories.txt file to build additional projects linking to it.")
endif (EXISTS subdirectories.txt)
