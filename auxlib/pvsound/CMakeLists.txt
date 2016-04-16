include(PVAddLibrary)

# Name the target based on the directory name
get_filename_component(TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/Modules")

find_package(LIBSNDFILE)

# Add the executable
pv_add_library(${TARGET} SRC
  src/NewCochlear.cpp
  src/NewCochlear.h
  src/PVsoundRegisterKeywords.cpp
  src/PVsoundRegisterKeywords.hpp
  src/SoundStream.cpp
  src/SoundStream.hpp
)

include_directories(${TARGET} ${LIBSNDFILE_INCLUDE_DIR})

target_link_libraries(${TARGET} ${LIBSNDFILE_LIBRARY})

# Dependencies
add_dependencies(${TARGET} pv)
