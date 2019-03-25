include(PVAddExecutable)

set(PV_PROJECT_NAME compareparams)

#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
set(${PV_PROJECT_NAME}_SRCCPP
  compareparams.cpp
)

pv_add_executable(${PV_PROJECT_NAME}
  SRC ${${PV_PROJECT_NAME}_SRCCPP}
)

pv_add_executable(readpvpheader SRC readpvpheader.c)

add_dependencies(${PV_PROJECT_NAME} pv)
