# Try and find cudNN
#
# Once done, this will define
#
#  CUDNN_FOUND - system has CUDNN
#  CUDNN_INCLUDE_DIRS - the CUDNN include directories
#  CUDNN_LIBRARIES - link these to use cudNN
#
# A CUDNN_PATH variable can be set prior find_package(CUDNN)
# to specify exactly where to search


include(LibFindMacros)

foreach(PATH_HINT ${CUDNN_PATH})
  set(CUDNN_HEADER_SEARCH_PATHS ${CUDNN_HEADER_SEARCH_PATHS};${CUDNN_PATH}/include)
  set(CUDNN_LIBRARY_SEARCH_PATHS ${CUDNN_LIBRARY_SEARCH_PATHS};${CUDNN_PATH}/lib)
  set(CUDNN_LIBRARY_SEARCH_PATHS ${CUDNN_LIBRARY_SEARCH_PATHS};${CUDNN_PATH}/lib64)

  if(CUDA_FOUND)
    set(CUDNN_HEADER_SEARCH_PATHS ${CUDNN_HEADER_SEARCH_PATHS};${CUDNN_PATH}/include)
    set(CUDNN_HEADER_SEARCH_PATHS ${CUDNN_HEADER_SEARCH_PATH};${CUDA_TOOLKIT_ROOT_DIR}/include;${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include)
    set(CUDNN_LIBRARY_SEARCH_PATHS ${CUDNN_LIBRARY_SEARCH_PATH};${CUDA_TOOLKIT_ROOT_DIR}/lib64;${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
  endif()
endforeach()

foreach(PATH_HINT ${CUDNN_PATH})
  set(CUDNN_HEADER_SEARCH_PATHS ${CUDNN_HEADER_SEARCH_PATHS};${CUDNN_PATH}/include)
  set(CUDNN_LIBRARY_SEARCH_PATHS ${CUDNN_LIBRARY_SEARCH_PATHS};${CUDNN_PATH}/lib)
  set(CUDNN_LIBRARY_SEARCH_PATHS ${CUDNN_LIBRARY_SEARCH_PATHS};${CUDNN_PATH}/lib64)
endforeach()

if (APPLE)
  set(CUDNN_HEADER_SEARCH_PATHS
    ${CUDNN_HEADER_SEARCH_PATHS}
    /usr/local/cudnn-7.0/include
    /usr/local/cudnn-6.5/include
    )

  set(CUDNN_LIBRARY_SEARCH_PATHS
    ${CUDNN_LIBRARY_SEARCH_PATHS}
    /usr/local/cudnn-7.0/lib
    /usr/local/cudnn-7.0/lib64
    /usr/local/cudnn-6.5/lib
    /usr/local/cudnn-6.5/lib64
    )

else()
  set(CUDNN_HEADER_SEARCH_PATHS
    ${CUDNN_HEADER_SEARCH_PATHS}
    /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v4/include
    /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v3/include
    /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/include
    )
  set(CUDNN_LIBRARY_SEARCH_PATHS
    ${CUDNN_LIBRARY_SEARCH_PATHS}
    /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v4/lib
    /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v4/lib64
    /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v3/lib
    /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v3/lib64
    /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/lib
    /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/lib64
    )
endif()

find_path(CUDNN_INCLUDE_DIR
  NAMES cudnn.h
  PATHS ${CUDNN_HEADER_SEARCH_PATHS}
  DOC "cuDNN include"
  )

# Find cudnn library
find_library(CUDNN_LIBRARY cudnn cudnn_static ${CUDNN_LIBRARY_SEARCH_PATHS} DOC "cuDNN library")

set(CUDNN_PROCESS_INCLUDES CUDNN_INCLUDE_DIR)
set(CUDNN_PROCESS_LIBS CUDNN_LIBRARY)
libfind_process(CUDNN)
