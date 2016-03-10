################################################################################
# Default values
################################################################################

macro(pv_config_project)

  # The default build type if CMAKE_BUILD_TYPE is empty
  set(CMAKE_DEFAULT_BUILD_TYPE "Release")

  # Intel Compiler defaults
  set(ICC_OPT_REPORT_LEVEL "-qopt-report=2")
  set(ICC_OPT_REPORT_PHASE "-opt-report-phase=loop,vec,openmp")
  set(ICC_DEBUG_FLAGS -restrict;-traceback;-xcore-avx2;${ICC_OPT_REPORT_LEVEL};${ICC_OPT_REPORT_PHASE};-g;-O2)
  set(ICC_RELEASE_FLAGS -restrict;-traceback;-xcore-avx2;-O3;-DNDEBUG)
  set(ICC_OPENMP_FLAG "-qopenmp")
  set(ICC_CC "icc")
  set(ICC_CXX "icpc")
  set(ICC_SANITIZE_ADDRESS_CXX_FLAGS "")
  set(ICC_SANITIZE_ADDRESS_EXE_LINKER_FLAGS "")
  
  # Clang Compiler defaults
  set(CLANG_OPENMP_FLAG "-fopenmp=libiomp5")
  set(CLANG_SANITIZE_ADDRESS_CXX_FLAGS "-g -fsanitize=address -fno-omit-frame-pointer")
  set(CLANG_SANITIZE_ADDRESS_LINKER_FLAGS -g;-fsanitize=address)
  # Flag to pass in to NVCC (which in turn passes this on to clang) so that off_t is defined
  set(CLANG_NVCC_FLAGS "-stdlib=libstdc++")
  set(CLANG_RELEASE_FLAGS "")
  
  # GCC compiler defaults
  set(GCC_OPENMP_FLAG "-fopenmp")
  set(GCC_SANITIZE_ADDRESS_CXX_FLAGS -g;-fsanitize=address;-fno-omit-frame-pointer)
  set(GCC_SANITIZE_ADDRESS_LINKER_FLAGS -g;-fsanitize=address)
  set(GCC_RELEASE_FLAGS "")
  set(GCC_LINK_LIBRARIES m)
  
  # CUDA flags
  set(CUDA_BASE_FLAGS "-arch=sm_30")
  set(CUDA_RELEASE_FLAGS "${CUDA_BASE_FLAGS};-O")
  set(CUDA_DEBUG_FLAGS "${CUDA_BASE_FLAGS};-Xptxas;-v;-keep;-lineinfo;-g;-G")
  
  # CUDNN path hints
  set(APPLE_CUDNN_PATH_HINT "/usr/local/cudnn-7.0" "/usr/local/cudnn")
  set(LINUX_CUDNN_PATH_HINT "/nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v4/" "/cuDNN/cudnn-7.0-linux-x64-v4" "/usr/local/cudnn")
  
  # Help strings
  set(PV_DIR_HELP "The core PetaVision directory")
  set(PV_OPENMP_HELP "Defines if PetaVision uses OpenMP")
  set(PV_OPENMP_FLAG_HELP "Compiler flag for compiling with OpenMP")
  set(PV_USE_MPI_HELP "Defines whether PetaVision uses MPI")
  set(PV_USE_CUDA_HELP "Defines if PetaVision uses CUDA GPU")
  set(PV_CUDA_RELEASE_HELP "Defines if Cuda compiles with optimization")
  set(PV_USE_GDAL_HELP "Enable loading of images and movies")
  set(PV_CUDNN_PATH_HELP "Location of cuDNN libraries. Optional")
  set(PV_ADDRESS_SANITIZE_HELP "Add compiler flags for sanitizing addresses")
  set(PV_BUILD_SHARED_HELP "Build a shared library")
  
  ################################################################################
  # Detect the compiler and OpenMP capabilities
  ################################################################################
  
  # Set defaults based on the detected compiler
  set(COMPILER_DETECTED OFF)
  set(INTEL_COMPILER_DETECTED OFF)
  
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    # Intel compiler detected
    set(COMPILER_DETECTED ON)
    set(INTEL_COMPILER_DETECTED ON)
    message(STATUS "Intel compiler detected")
    set(PV_USE_OPENMP ON CACHE BOOL "${PV_USE_OPENMP_HELP}")
    set(PV_OPENMP_FLAG ${ICC_OPENMP_FLAG} CACHE STRING "${PV_OPENMP_FLAG_HELP}")
    set(PV_SANITIZE_ADDRESS_CXX_FLAGS "${ICC_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_SANITIZE_ADDRESS_LINKER_FLAGS "${ICC_SANITIZE_ADDRESS_LINKER_FLAGS}")
    set(PV_COMPILE_FLAGS_DEBUG ${ICC_DEBUG_FLAGS})
    set(PV_COMPILE_FLAGS_RELEASE ${ICC_RELEASE_FLAGS})
    # NVCC isn't compatible with the Intel compiler
    set(PV_USE_CUDA OFF CACHE BOOL "${PV_USE_CUDA_HELP}")
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    # Clang detected
    set(PV_SANITIZE_ADDRESS_CXX_FLAGS "${CLANG_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_SANITIZE_ADDRESS_LINKER_FLAGS "${CLANG_SANITIZE_ADDRESS_LINKER_FLAGS}")
    list(APPEND PV_COMPILE_FLAGS_DEBUG ${CLANG_COMPILE_FLAGS_DEBUG})
    list(APPEND PV_COMPILE_FLAGS_RELEASE ${CLANG_COMPILE_FLAGS_RELEASE})
    
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER "7.0")
      # Xcode detected
      # Apple's Xcode clang compiler detected, which has a greater version number than
      # open source clang. This test for Apple's clang compiler that does not support
      # OpenMP will eventually fail when open source clang version numbers catch up
      # to Xcode version numbers
      set(COMPILER_DETECTED ON)
      message(STATUS "Xcode clang compiler detected. No OpenMP support.")
      set(PV_USE_OPENMP OFF CACHE BOOL "${PV_USE_OPENMP_HELP}")
    else()
      # Open source clang detected
      set(COMPILER_DETECTED ON)
      set(PV_NVCC_FLAGS ${CLANG_NVCC_FLAGS})
      if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER "3.5.0" OR ${CMAKE_CXX_COMPILER_VERSION} VERSION_EQUAL "3.5.0")
        # This is a clang that supports OpenMP. Note how Apple's clang has a version
        # number greater than this. What could go wrong?
        message(STATUS "Clang with OpenMP support detected")
        set(PV_USE_OPENMP ON CACHE BOOL "${PV_USE_OPENMP_HELP}")
        set(PV_OPENMP_FLAG ${CLANG_OPENMP_FLAG} CACHE STRING "${PV_OPENMP_FLAG_HELP}")
        set(PV_OPENMP_LIBRARIES "/usr/local/lib/libiomp5.dylib")
      else()
        # Clang detected that does not support OpenMP
        message(STATUS "Clang without OpenMP support detected")
        set(PV_USE_OPENMP OFF CACHE BOOL "${PV_USE_OPENMP_HELP}")
        set(PV_OPENMP_FLAG "" CACHE STRING "${PV_OPENMP_FLAG_HELP}")
      endif()
    endif()
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # GCC detected
    set(COMPILER_DETECTED ON)
    set(PV_SANITIZE_ADDRESS_CXX_FLAGS "${GCC_SANITIZE_ADDRESS_CXX_FLAGS}")
    set(PV_SANITIZE_ADDRESS_LINKER_FLAGS "${GCC_SANITIZE_ADDRESS_LINKER_FLAGS}")
    set(PV_COMPILE_FLAGS_DEBUG ${GCC_COMPILE_FLAGS_DEBUG})
    set(PV_COMPILE_FLAGS_RELEASE ${GCC_COMPILE_FLAGS_RELEASE})
    set(PV_LINK_LIBRARIES ${GCC_LINK_LIBRARIES})
    
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
    message(WARNING "Compiler not detected. Your life just got a little bit harder. Bummer.")
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
  set(PV_CUDA_RELEASE ON CACHE BOOL ${PV_CUDA_RELEASE_HELP})
  set(PV_USE_GDAL ON CACHE BOOL "${PV_USE_GDAL_HELP}")
  set(PV_ADDRESS_SANITIZE OFF CACHE BOOL "${PV_ADDRESS_SANITIZE_HELP}")
  set(PV_BUILD_SHARED OFF CACHE BOOL "{$PV_BUILD_SHARED_HELP}")
  
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
    set(CMAKE_BUILD_TYPE "${CMAKE_DEFAULT_BUILD_TYPE}")
  endif()
  
  # Set up PV_SOURCE and INCLUDE_DIR.
  set(PV_SOURCE_DIR "${PV_DIR}/src")
  set(PV_INCLUDE_DIR "${PV_DIR}/src")
  
  set(PV_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/pv-core/lib")
  
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
      if(APPLE)
        set(CUDNN_PATH ${APPLE_CUDNN_PATH_HINT} CACHE PATH "${PV_CUDNN_PATH_HELP}")
      else()
        set(CUDNN_PATH ${LINUX_CUDNN_PATH_HINT} CACHE PATH "${PV_CUDNN_PATH_HELP}")
      endif()
      
      find_package(CUDA)
      find_package(CUDNN)
      
      # Set cuda compile flags
      if(CUDA_FOUND AND CUDNN_FOUND)
        # Used later to set an variable in cMakeHeader.h PV_USE_CUDNN.
        # Without this, none of the CUDNN code will be compiled and all
        # CUDA code will fail
        set(PV_USE_CUDNN ON)
        
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
  
  if (PV_USE_GDAL)
    find_package(GDAL)
  endif()
  
  if (PV_USE_MPI)
    find_package(MPI)
  endif()
endmacro()
