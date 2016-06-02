include(PVAddExecutable)

get_filename_component(PV_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

set(${PV_PROJECT_NAME}_SRCCPP
  src/AudioGroupHandler.cpp
  src/AudioInputLayer.cpp
  src/main.cpp
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib")

pv_add_executable(${PV_PROJECT_NAME}
  SRC ${${PV_PROJECT_NAME}_SRCCPP}
  #  OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}"
)

target_link_libraries(${PV_PROJECT_NAME} Aquila Ooura_fft)

add_dependencies(${PV_PROJECT_NAME} pv)
