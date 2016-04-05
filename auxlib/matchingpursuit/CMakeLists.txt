# Adds the matchingpursuit library, which provides PetaVision layer classes
# for performing the matching pursuit algorithm in a PetaVision column.
#
# To link a project to this library, add the lines
#
# target_link_libraries(${BASE_NAME} matchingpursuit)
# add_dependencies(${BASE_NAME} matchingpursuit)
#
# to the project's CMakeLists.txt file, and
#
# add the line
#
# auxlib/matchingpursuit
#
# to the subdirectories.txt file in your CMake build folder.


include(PVAddLibrary)

get_filename_component(PV_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(${PV_PROJECT_NAME}_LIBRARIES_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)

set(${PV_PROJECT_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(LIBRARY_OUTPUT_PATH ${${PV_PROJECT_NAME}_LIBRARIES_DIR})

include_directories("${${PV_PROJECT_NAME}_SOURCE_DIR}")

set(${PV_PROJECT_NAME}_SRCCPP
  src/MatchingPursuitGroupHandler.cpp
  src/MatchingPursuitGroupHandler.hpp
  src/MatchingPursuitLayer.cpp
  src/MatchingPursuitLayer.hpp
  src/MatchingPursuitResidual.cpp
  src/MatchingPursuitResidual.hpp
)

pv_add_library(${PV_PROJECT_NAME}
  SRC ${${PV_PROJECT_NAME}_SRCCPP}
  OUTPUT_PATH "${${PV_PROJECT_NAME}_LIBRARIES_DIR}"
)

add_dependencies(${PV_PROJECT_NAME} pv)

set_target_properties(${PV_PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH})
set_target_properties(${PV_PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG ${LIBRARY_OUTPUT_PATH})
set_target_properties(${PV_PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
set_target_properties(${PV_PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH})
set_target_properties(${PV_PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${LIBRARY_OUTPUT_PATH})
set_target_properties(${PV_PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
