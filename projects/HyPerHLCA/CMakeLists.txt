include(PVAddExecutable)
# Name the target based on the directory name
get_filename_component(TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add the executable
pv_add_executable(${TARGET} SRC src/main.cpp)

# This target depends on libpv being built
add_dependencies(${TARGET} pv)
