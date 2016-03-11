include(PVAddExecutable)
# Name the target based on the directory name
get_filename_component(TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add the executable
pv_add_executable(${TARGET} SRC
  src/BIDS_main_buildandrun.cpp
  src/BIDSCloneLayer.cpp
  src/BIDSConn.cpp
  src/BIDSGroupHandler.cpp
  src/BIDSLayer.cpp
  src/BIDSMovieCloneMap.cpp
  src/BIDSSensorLayer.cpp
  src/InitBIDSLateral.cpp
  src/InitBIDSLateralParams.cpp
  src/BIDSCloneLayer.hpp
  src/BIDSConn.hpp
  src/BIDSGroupHandler.hpp
  src/BIDSLayer.hpp
  src/BIDSMovieCloneMap.hpp
  src/BIDSSensorLayer.hpp
  src/InitBIDSLateral.hpp
  src/InitBIDSLateralParams.hpp
)

# This target depends on libpv being built
add_dependencies(${TARGET} pv)
