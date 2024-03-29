################################################################################
# Default values
################################################################################

macro(pv_config_project)

  # set policy to use CUDA_ROOT and CUDNN_ROOT variables
  if (${CMAKE_VERSION} VERSION_GREATER "3.16.0")
    cmake_policy(SET CMP0074 NEW)
  endif()

  # The default build type if CMAKE_BUILD_TYPE is empty
  set(PV_DEFAULT_BUILD_TYPE "Release")

  # Intel Compiler defaults
  set(ICC_OPT_REPORT_LEVEL "-qopt-report=5")
  set(ICC_OPT_REPORT_PHASE "-qopt-report-phase=all")
  set(ICC_DEBUG_FLAGS -restrict;-traceback;${ICC_OPT_REPORT_LEVEL};${ICC_OPT_REPORT_PHASE};-g;-O2;-Winline)
  set(ICC_RELEASE_FLAGS -restrict;-traceback;${ICC_OPT_REPORT_LEVEL};${ICC_OPT_REPORT_PHASE};-O3;-DNDEBUG;-qopt-zmm-usage=high;-Winline)
  set(ICC_OPENMP_FLAG "-qopenmp")
  set(ICC_SANITIZE_ADDRESS_CXX_FLAGS "")
  set(ICC_SANITIZE_ADDRESS_EXE_LINKER_FLAGS "")
  set(ICC_CPP_11X_FLAGS "")

  # Intel LLVM-based Compiler defaults
  set(INTELLLVM_DEBUG_FLAGS -g;-O2;-Winline)
  set(INTELLLVM_RELEASE_FLAGS -O3;-DNDEBUG;-Winline)
  set(INTELLLVM_OPENMP_FLAG "-qopenmp")
  set(INTELLLVM_SANITIZE_ADDRESS_CXX_FLAGS "")
  set(INTELLLVM_SANITIZE_ADDRESS_EXE_LINKER_FLAGS "")
  set(INTELLLVM_CPP_11X_FLAGS "")

  # LLVM Clang Compiler defaults
  set(CLANG_DEBUG_FLAGS -g;-O2)
  set(CLANG_RELEASE_FLAGS -O3;-DNDEBUG)
  set(CLANG_OPENMP_FLAG "-fopenmp")
  set(CLANG_SANITIZE_ADDRESS_CXX_FLAGS "")
  set(CLANG_SANITIZE_ADDRESS_EXE_LINKER_FLAGS "")
  set(CLANG_CPP_11X_FLAGS "")

  # AppleClang Compiler defaults
  set(APPLECLANG_DEBUG_FLAGS "-g;-O2")
  set(APPLECLANG_RELEASE_FLAGS "-O3;-DNDEBUG")
  set(APPLECLANG_CPP_11X_FLAGS "-std=c++11 -stdlib=libc++")
  set(APPLECLANG_SANITIZE_ADDRESS_CXX_FLAGS "-g -fsanitize=address -fno-omit-frame-pointer")
  set(APPLECLANG_SANITIZE_ADDRESS_LINKER_FLAGS -g;-fsanitize=address)
  # Flag to pass in to NVCC (which in turn passes this on to clang) so that off_t is defined
  set(APPLECLANG_NVCC_FLAGS "-stdlib=libstdc++")
  
  # GCC compiler defaults
  set(GCC_OPENMP_FLAG "-fopenmp")
  # warning flag is here so that it skips nvcc, which causes inaccurate warnings by compiling .cu as C
  #  set(GCC_CPP_11X_FLAGS "-std=c++11 -Wall -fdiagnostics-show-option")
  # Warnings disabled so that they can be addressed in a seperate branch
  set(GCC_CPP_11X_FLAGS "-std=c++11")
  set(GCC_SANITIZE_ADDRESS_CXX_FLAGS -g;-fsanitize=address;-fno-omit-frame-pointer)
  set(GCC_SANITIZE_ADDRESS_LINKER_FLAGS -g;-fsanitize=address)
  set(GCC_COMPILE_FLAGS_DEBUG -Wdouble-promotion -Wreturn-type)
  set(GCC_RELEASE_FLAGS "")
  set(GCC_LINK_LIBRARIES m)
  
  # Help strings
  set(PV_DIR_HELP "The core PetaVision directory")
  set(PV_OPENMP_HELP "Defines if PetaVision uses OpenMP")
  set(PV_OPENMP_FLAG_HELP "Compiler flag for compiling with OpenMP")
  set(PV_USE_MPI_HELP "Defines whether PetaVision uses MPI")
  set(PV_USE_CUDA_HELP "Defines if PetaVision uses CUDA GPU")
  set(PV_CUDA_RELEASE_HELP "Defines if CUDA compiles with optimization")
  set(PV_CUDA_ARCHITECTURE_HELP "Defines CUDA architecture to compile for")
  set(PV_USE_LUA_HELP "Enable using a lua program as the params file")
  set(PV_CUDNN_PATH_HELP "Location of cuDNN libraries. Optional")
  set(PV_ADDRESS_SANITIZE_HELP "Add compiler flags for sanitizing addresses")
  set(PV_BUILD_SHARED_HELP "Build a shared library")
  set(PV_DEBUG_OUTPUT_HELP "Display output from logDebug() in Release builds")
  set(PV_TIMER_VERBOSE_HELP "Print a message whenever a Timer object starts or stops")
  set(PV_BUILD_TEST_HELP "Build the OpenPV test suite")
  set(PV_COMPILE_OPTIONS_EXTRA_HELP "Any additional flags to pass to the compiler")

  ################################################################################
  # Detect the compiler and OpenMP capabilities
  ################################################################################
  
  # Set defaults based on the detected compiler
  set(COMPILER_DETECTED OFF)
  set(INTEL_COMPILER_DETECTED OFF)
  set(APPLECLANG_COMPILER_DETECTED OFF)

  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    # Intel-classic compiler detected
    set(COMPILER_DETECTED ON)
    set(INTEL_COMPILER_DETECTED ON)
    message(STATUS "Intel classic compiler detected")
    set(PV_USE_OPENMP ON CACHE BOOL "${PV_USE_OPENMP_HELP}")
    set(PV_OPENMP_FLAG ${ICC_OPENMP_FLAG} CACHE STRING "${PV_OPENMP_FLAG_HELP}")
    set(PV_SANITIZE_ADDRESS_CXX_FLAGS "${ICC_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_SANITIZE_ADDRESS_LINKER_FLAGS "${ICC_SANITIZE_ADDRESS_LINKER_FLAGS}")
    set(PV_COMPILE_FLAGS_DEBUG ${ICC_DEBUG_FLAGS})
    set(PV_COMPILE_FLAGS_RELEASE ${ICC_RELEASE_FLAGS})
    set(PV_CPP_11X_FLAGS ${ICC_CPP_11X_FLAGS})
    # NVCC isn't compatible with the Intel compiler
    set(PV_USE_CUDA OFF CACHE BOOL "${PV_USE_CUDA_HELP}")
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
    # Intel-classic compiler detected
    set(COMPILER_DETECTED ON)
    set(INTEL_COMPILER_DETECTED ON)
    message(STATUS "Intel oneAPI compiler detected")
    set(PV_USE_OPENMP ON CACHE BOOL "${PV_USE_OPENMP_HELP}")
    set(PV_OPENMP_FLAG ${INTELLLVM_OPENMP_FLAG} CACHE STRING "${PV_OPENMP_FLAG_HELP}")
    set(PV_SANITIZE_ADDRESS_CXX_FLAGS "${INTELLLVM_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_SANITIZE_ADDRESS_LINKER_FLAGS "${INTELLLVM_SANITIZE_ADDRESS_LINKER_FLAGS}")
    set(PV_COMPILE_FLAGS_DEBUG ${INTELLLVM_DEBUG_FLAGS})
    set(PV_COMPILE_FLAGS_RELEASE ${INTELLLVM_RELEASE_FLAGS})
    set(PV_CPP_11X_FLAGS ${INTELLLVM_CPP_11X_FLAGS})
    # NVCC isn't compatible with the Intel compiler
    set(PV_USE_CUDA OFF CACHE BOOL "${PV_USE_CUDA_HELP}")
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    # LLVM Clang compiler detected
    set(COMPILER_DETECTED ON)
    set(LLVMCLANG_COMPILER_DETECTED ON)
    message(STATUS "LLVM Clang compiler detected")
    set(PV_USE_OPENMP ON CACHE BOOL "${PV_USE_OPENMP_HELP}")
    set(PV_OPENMP_FLAG ${CLANG_OPENMP_FLAG} CACHE STRING "${PV_OPENMP_FLAG_HELP}")
    set(PV_SANITIZE_ADDRESS_CXX_FLAGS "${CLANG_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_SANITIZE_ADDRESS_LINKER_FLAGS "${CLANG_SANITIZE_ADDRESS_LINKER_FLAGS}")
    set(PV_COMPILE_FLAGS_DEBUG ${CLANG_DEBUG_FLAGS})
    set(PV_COMPILE_FLAGS_RELEASE ${CLANG_RELEASE_FLAGS})
    set(PV_CPP_11X_FLAGS ${CLANG_CPP_11X_FLAGS})
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
    # Apple Clang detected
    set(COMPILER_DETECTED ON)
    set(APPLECLANG_COMPILER_DETECTED ON)
    message(STATUS "Apple Clang compiler detected")
    set(PV_USE_OPENMP OFF CACHE BOOL "${PV_USE_OPENMP_HELP}")
    set(PV_NVCC_FLAGS ${APPLECLANG_NVCC_FLAGS})
    set(PV_SANITIZE_ADDRESS_CXX_FLAGS "${APPLECLANG_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_SANITIZE_ADDRESS_LINKER_FLAGS "${APPLECLANG_SANITIZE_ADDRESS_LINKER_FLAGS}")
    set(PV_CPP_11X_FLAGS "${APPLECLANG_CPP_11X_FLAGS}")
    list(APPEND PV_COMPILE_FLAGS_DEBUG ${APPLECLANG_DEBUG_FLAGS})
    list(APPEND PV_COMPILE_FLAGS_RELEASE ${APPLECLANG_RELEASE_FLAGS})
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # GCC detected
    set(COMPILER_DETECTED ON)
    set(PV_SANITIZE_ADDRESS_CXX_FLAGS "${GCC_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_SANITIZE_ADDRESS_LINKER_FLAGS "${GCC_SANITIZE_ADDRESS_LINKER_FLAGS}")
    set(PV_COMPILE_FLAGS_DEBUG ${GCC_COMPILE_FLAGS_DEBUG})
    set(PV_COMPILE_FLAGS_RELEASE ${GCC_COMPILE_FLAGS_RELEASE})
    set(PV_LINK_LIBRARIES ${GCC_LINK_LIBRARIES})
    set(PV_CPP_11X_FLAGS ${GCC_CPP_11X_FLAGS})
    
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER "4.2")
      set(PV_USE_OPENMP ON CACHE BOOL "${PV_USE_OPENMP_HELP}")
      set(PV_OPENMP_FLAG ${GCC_OPENMP_FLAG} CACHE STRING "${PV_OPENMP_FLAG_HELP}")
    else ()
      set(PV_USE_OPENMP OFF CACHE BOOL "${PV_USE_OPENMP_HELP}")
      set(PV_OPENMP_FLAG "" CACHE STRING "${PV_OPENMP_FLAG_HELP}")
    endif()
  endif()
  
  # Compiler not detected. Bummer.  
  if (NOT COMPILER_DETECTED)
    message(WARNING "Compiler not recognized (\"${CMAKE_CXX_COMPILER_ID}\"). Your life just got a little bit harder. Bummer.")
    message(WARNING "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
    message(WARNING "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
  
  ################################################################################
  # Set/get cached variables
  ################################################################################
  set(PV_DIR ${PV_DIR_DEFAULT} CACHE PATH "${PV_DIR_HELP}")
  set(PV_USE_MPI ON CACHE BOOL "${PV_USE_MPI_HELP}")
  set(PV_USE_OPENMP ON CACHE BOOL "${PV_USE_OPENMP_HELP}")
  set(PV_OPENMP_FLAG "${PV_OPENMP_FLAG}" CACHE STRING "${PV_OPENMP_FLAG_HELP}")
  set(PV_USE_CUDA ON CACHE BOOL "${PV_USE_CUDA_HELP}")
  set(PV_CUDA_RELEASE ON CACHE BOOL "${PV_CUDA_RELEASE_HELP}")
  set(PV_CUDA_ARCHITECTURE "Auto" CACHE STRING "${PV_CUDA_ARCHITECTURE_HELP}")
  set(PV_USE_LUA OFF CACHE BOOL "${PV_USE_LUA_HELP}")
  set(PV_ADDRESS_SANITIZE OFF CACHE BOOL "${PV_ADDRESS_SANITIZE_HELP}")
  set(PV_BUILD_SHARED OFF CACHE BOOL "${PV_BUILD_SHARED_HELP}")
  set(PV_DEBUG_OUTPUT OFF CACHE BOOL "${PV_DEBUG_OUTPUT_HELP}")
  set(PV_TIMER_VERBOSE OFF CACHE BOOL "${PV_TIMER_VERBOSE_HELP}")
  set(PV_BUILD_TEST ON CACHE BOOL "${PV_BUILD_TEST_HELP}")
  set(PV_COMPILE_OPTIONS_EXTRA "" CACHE STRING "${PV_COMPILE_OPTIONS_EXTRA_HELP}")

  ################################################################################
  # Set compiler flags
  ################################################################################
  
  if (PV_USE_OPENMP)
    message(STATUS "OpenMP support enabled")
    list(APPEND PV_COMPILE_FLAGS_DEBUG ${PV_OPENMP_FLAG})
    list(APPEND PV_COMPILE_FLAGS_RELEASE ${PV_OPENMP_FLAG})
    set(PV_LINKER_FLAGS ${PV_LINKER_FLAGS};${PV_OPENMP_FLAG})
    if (PV_OPENMP_LIBRARY)
      set(PV_LINKER_FLAGS ${PV_LINKER_FLAGS};${PV_OPENMP_LIBRARY})
    endif()
  endif()
  
  if (PV_ADDRESS_SANITIZE)
    set(PV_COMPILE_FLAGS_DEBUG "${PV_COMPILE_FLAGS_DEBUG} ${PV_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_COMPILE_FLAGS_RELEASE "${PV_COMPILE_FLAGS_RELEASE} ${PV_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_LINKER_FLAGS ${PV_LINKER_FLAGS};${PV_SANITIZE_ADDRESS_LINKER_FLAGS})
  endif()
  
  ################################################################################
  ################################################################################
  
  # Ensure that CMAKE_BUILD_TYPE is set
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "${PV_DEFAULT_BUILD_TYPE}")
  endif()
  
  # Set up PV_SOURCE and INCLUDE_DIR.
  set(PV_SOURCE_DIR ${PV_DIR})
  set(PV_INCLUDE_DIR ${PV_DIR})
  
  set(PV_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
  
  if (PV_BUILD_SHARED)
    if (APPLE)
      set(PV_LIBRARIES "${PV_LIBRARY_DIR}/libpv.dylib")
    else()
      set(PV_LIBRARIES "${PV_LIBRARY_DIR}/libpv.so")
    endif()
  else()
    set(PV_LIBRARIES "${PV_LIBRARY_DIR}/libpv.a")
  endif()
  
  set(PV_CONFIG_FILE_DIR "${PV_LIBRARY_DIR}/include")
  
  ################################################################################
  # Find supporting libraries
  ################################################################################
  if (PV_USE_CUDA)
    if(INTEL_COMPILER_DETECTED)
      message(WARNING "-- CUDA cannot be used with the Intel compiler. Disabling CUDA build")
      set(PV_USE_CUDA OFF)
    else()
  
      set(CUDNN_PATH_HINT "/usr/local/cudnn")
      set(CUDNN_PATH ${CUDNN_PATH_HINT} CACHE PATH "${PV_CUDNN_PATH_HELP}")
      
      find_package(CUDA)
      find_package(CUDNN)
      
      # Set cuda compile flags
      if(CUDA_FOUND AND CUDNN_FOUND)
        # Used later to set an variable in cMakeHeader.h PV_USE_CUDNN.
        # Without this, none of the CUDNN code will be compiled and all
        # CUDA code will fail
        set(PV_USE_CUDNN ON)

        if(NOT "${PV_CUDA_ARCHITECTURE}" STREQUAL "")
          cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS "${PV_CUDA_ARCHITECTURE}")
          set(CUDA_BASE_FLAGS "${CUDA_BASE_FLAGS};${CUDA_ARCH_FLAGS}")
        endif(NOT "${PV_CUDA_ARCHITECTURE}" STREQUAL "")
        # set(CUDA_BASE_FLAGS "-arch=sm_30 -std=c++11") # sm_30 is obsolete as of CUDA version 11.0.
        set(CUDA_BASE_FLAGS "${CUDA_BASE_FLAGS};-std=c++11")
        set(CUDA_RELEASE_FLAGS "${CUDA_BASE_FLAGS};-O3")
        set(CUDA_DEBUG_FLAGS "${CUDA_BASE_FLAGS};-Xptxas;-v;-keep;-g;--generate-line-info")
        
        if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinRelSize")
          set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_RELEASE_FLAGS}")
        else()
          set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_DEBUG_FLAGS}")
        endif()
      else()
        set(PV_USE_CUDA OFF)
      endif()
      
    endif()
  endif()
  
  if (PV_USE_MPI)
    find_package(MPI)
  endif()

  if (PV_USE_LUA)
    find_package(Lua)
    if (LUA_FOUND)
      if ((${LUA_VERSION_MAJOR} LESS 5) OR (${LUA_VERSION_MAJOR} EQUAL 5 AND ${LUA_VERSION_MINOR} LESS 2))
        message("PV_USE_LUA requires Lua version 5.2 or later; however, Lua version that was found is ${LUA_VERSION_STRING}")
      endif ((${LUA_VERSION_MAJOR} LESS 5) OR (${LUA_VERSION_MAJOR} EQUAL 5 AND ${LUA_VERSION_MINOR} LESS 2))
    else (LUA_FOUND)
      message(FATAL_ERROR "Lua was not found")
    endif (LUA_FOUND)
  endif (PV_USE_LUA)
endmacro()
