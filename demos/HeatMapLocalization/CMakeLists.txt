include(PVAddExecutable)

get_filename_component(PV_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(${GDAL_INCLUDE_DIR})
set(${PV_PROJECT_NAME}_SRCCPP
  src/BBFind.cpp
  src/BBFindConfRemapLayer.cpp
  src/BBFindConfRemapProbe.cpp
  src/ConvertFromTable.cpp
  src/LocalizationProbe.cpp
  src/LocalizationBBFindProbe.cpp
  src/main.cpp
)

pv_add_executable(${PV_PROJECT_NAME}
  SRC ${${PV_PROJECT_NAME}_SRCCPP}
  #  OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}"
)

add_dependencies(${PV_PROJECT_NAME} pv)

find_package(GDAL)
if(GDAL_FOUND)
   target_link_libraries(${PV_PROJECT_NAME} ${GDAL_LIBRARIES})
else()
   message("ERROR: HeatMapLocalization requires GDAL.")
endif()

