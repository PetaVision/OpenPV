include(PVAddExecutable)

set(PV_PROJECT_NAME compareparams)

pv_add_executable(compareparams SRC compareparams.cpp)
add_dependencies(compareparams pv)

pv_add_executable(readpvpheader SRC readpvpheader.c)
add_dependencies(readpvpheader pv)
