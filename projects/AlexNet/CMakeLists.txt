include(PVAddExecutable)
# Name the target based on the directory name
get_filename_component(TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(PVMLEARNING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../auxlib/mlearning" CACHE PATH "The base directory for the PV mlearning library")
#set(PVMLEARNING_BINARY_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/../auxlib/mlearning" CACHE PATH "The base directory for the PV mlearning library")

if(NOT EXISTS "${PVMLEARNING_SOURCE_DIR}")
   message(FATAL_ERROR "PVMLEARNING_SOURCE_DIR \"${PVMLEARNING_SOURCE_DIR}\" does not exist.")
endif()

set(PVMLEARNING_LIBRARY_DIR "${PVMLEARNING_BINARY_DIR}/lib")
#set(PVMLEARNING_SOURCE_DIR  "${PVMLEARNING_SOURCE_DIR}/src")

include_directories(${PVMLEARNING_SOURCE_DIR}/src)

# Add the executable
pv_add_executable(${TARGET} SRC
  src/CustomGroupHandler.cpp
  src/main.cpp
  src/ProbeLayer.cpp
  src/SoftMaxBackprop.cpp
  src/CIFARGTLayer.hpp
  src/CustomGroupHandler.hpp
  src/ProbeLayer.hpp
  src/SoftMaxBackprop.hpp
)

# This target depends on libpv being built
add_dependencies(${TARGET} pv)
add_dependencies(${TARGET} auxlib)
