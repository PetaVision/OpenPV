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
#
# Call FindCUDA before using FindCUDNN
#
# That will set the CUDA_FOUND variable and expand the search
# for cuDNN to the cuda directories


include(LibFindMacros)

foreach(PATH_HINT ${CUDNN_PATH})
  list(APPEND CUDNN_HEADER_SEARCH_PATHS ${CUDNN_PATH}/include)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS ${CUDNN_PATH}/lib64)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS ${CUDNN_PATH}/lib)

  if(CUDA_FOUND)
    list(APPEND CUDNN_HEADER_SEARCH_PATHS ${CUDNN_PATH}/include)
    list(APPEND CUDNN_HEADER_SEARCH_PATHS ${CUDA_TOOLKIT_ROOT_DIR}/include)
    list(APPEND CUDNN_HEADER_SEARCH_PATHS ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include)
    list(APPEND CUDNN_LIBRARY_SEARCH_PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    list(APPEND CUDNN_LIBRARY_SEARCH_PATHS ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
  endif()
endforeach()

if (APPLE)
  list(APPEND CUDNN_HEADER_SEARCH_PATHS /usr/local/cudnn-7.0/include)
  list(APPEND CUDNN_HEADER_SEARCH_PATHS /usr/local/cudnn-6.5/include)

  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /usr/local/cudnn-7.0/lib)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /usr/local/cudnn-7.0/lib64)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /usr/local/cudnn-6.5/lib)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /usr/local/cudnn-6.5/lib64)

else()
  list(APPEND CUDNN_HEADER_SEARCH_PATHS /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v4/include)
  list(APPEND CUDNN_HEADER_SEARCH_PATHS /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v3/include)
  list(APPEND CUDNN_HEADER_SEARCH_PATHS /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/include)

  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v4/lib)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v4/lib64)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v3/lib)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /nh/compneuro/Data/cuDNN/cudnn-7.0-linux-x64-v3/lib64)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/lib)
  list(APPEND CUDNN_LIBRARY_SEARCH_PATHS /nh/compneuro/Data/cuDNN/cudnn-6.5-linux-x64-R2-rc1/lib64)
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
