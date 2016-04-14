include(PVAddLibrary)

# Name the target based on the directory name
get_filename_component(TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add the executable
pv_add_library(${TARGET} SRC
  src/MatchingPursuitLayer.cpp
  src/MatchingPursuitLayer.hpp
  src/MatchingPursuitRegisterKeywords.cpp
  src/MatchingPursuitRegisterKeywords.hpp
  src/MatchingPursuitResidual.cpp
  src/MatchingPursuitResidual.hpp
)

# Dependencies
add_dependencies(${TARGET} pv)
