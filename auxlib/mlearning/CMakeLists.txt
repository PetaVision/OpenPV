include(PVAddLibrary)

# Name the target based on the directory name
get_filename_component(TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add the executable
pv_add_library(${TARGET} SRC
  src/GradientCheckConn.cpp
  src/MLearningGroupHandler.cpp
  src/MLPErrorLayer.cpp
  src/MLPForwardLayer.cpp
  src/MLPOutputLayer.cpp
  src/MLPSigmoidLayer.cpp
  src/GradientCheckConn.hpp
  src/MLearningGroupHandler.hpp
  src/MLPErrorLayer.hpp
  src/MLPForwardLayer.hpp
  src/MLPOutputLayer.hpp
  src/MLPSigmoidLayer.hpp
)

# Dependencies
add_dependencies(${TARGET} pv)
