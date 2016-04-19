include(PVAddExecutable)
# Name the target based on the directory name
get_filename_component(TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(PVSOUND_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../auxlib/pvsound/" CACHE PATH "The pvsound source directory")
set(PVSOUND_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../auxlib/pvsound/" CACHE PATH "The directory containing the pvsound library")

if(NOT EXISTS "${PVSOUND_SOURCE_DIR}")
   message(FATAL_ERROR "PVSOUND_SOURCE_DIR \"${PVSOUND_SOURCE_DIR}\" does not exist.")
endif()

set(PVSOUND_INCLUDE_DIR "${PVSOUND_SOURCE_DIR}/src")

include_directories(${TARGET} "${PVSOUND_INCLUDE_DIR}")
include_directories(${TARGET} ${LIBSNDFILE_INCLUDE_DIR})

# Add the executable
pv_add_executable(${TARGET} SRC
  src/CochlearLayer.cpp
  src/CochlearLayer.hpp
  src/inverseCochlearLayer.cpp
  src/inverseCochlearLayer.hpp
  src/inverseNewCochlearLayer.cpp
  src/inverseNewCochlearLayer.hpp
  src/main.cpp
  src/SoundProbe.cpp
  src/SoundProbe.hpp
  src/StreamReconLayer.cpp
  src/StreamReconLayer.h
  OUTPUT_PATH "${CMAKE_BUILD_TYPE}"
)

if (PV_BUILD_SHARED)
  if (APPLE)
    set(PVSOUND_LIBRARIES "${PVSOUND_BINARY_DIR}/libpvsound.dylib")
  else()
    set(PVSOUND_LIBRARIES "${PVSOUND_BINARY_DIR}/libpvsound.so")
  endif(APPLE)
else()
  set(PVSOUND_LIBRARIES "${PVSOUND_BINARY_DIR}/libpvsound.a")
endif(PV_BUILD_SHARED)

target_link_libraries(${TARGET} ${LIBSNDFILE_LIBRARY})
target_link_libraries(${TARGET} ${PVSOUND_LIBRARIES})

# This target depends on libpv and libpvsound being built
add_dependencies(${TARGET} pv)
add_dependencies(${TARGET} pvsound)

#Create a symlink of source tree's input directory into build tree.
if (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
  set(TEST_SOURCE_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/input")
  if (EXISTS "${TEST_SOURCE_INPUT}")
    set(TEST_BINARY_INPUT "${CMAKE_CURRENT_BINARY_DIR}/input")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${TEST_SOURCE_INPUT} ${TEST_BINARY_INPUT})
  endif (EXISTS "${TEST_SOURCE_INPUT}")
endif(NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})

