include(PVAddLibrary)

# Name the target based on the directory name
get_filename_component(TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add the executable
#TODO mirror this CMakeLists with the ones in pv-core.
#for now, we add all layers this way
pv_add_library(${TARGET} SRC
  src/connections/GradientCheckConn.cpp
  src/connections/GradientCheckConn.hpp
  src/columns/MLPRegisterKeywords.cpp
  src/columns/MLPRegisterKeywords.hpp
  src/layers/MLPErrorLayer.cpp
  src/layers/MLPErrorLayer.hpp
  src/layers/MLPForwardLayer.cpp
  src/layers/MLPForwardLayer.hpp
  src/layers/MLPOutputLayer.cpp
  src/layers/MLPOutputLayer.hpp
  src/layers/MLPSigmoidLayer.cpp
  src/layers/MLPSigmoidLayer.hpp
  #src/layers/BatchNorm.cpp
  #src/layers/BatchNorm.hpp
)

# Dependencies
add_dependencies(${TARGET} pv)
