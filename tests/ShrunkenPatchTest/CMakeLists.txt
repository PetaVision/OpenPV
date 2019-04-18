set(SRC_CPP
  src/pv.cpp
  src/ShrunkenPatchTestLayer.cpp
  src/ShrunkenPatchTestProbe.cpp
)

set(SRC_HPP
  src/ShrunkenPatchTestLayer.hpp
  src/ShrunkenPatchTestProbe.hpp
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
