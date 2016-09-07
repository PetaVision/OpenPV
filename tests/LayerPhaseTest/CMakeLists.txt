set(SRC_CPP
  src/LayerPhaseTest.cpp
  src/LayerPhaseTestProbe.cpp
)

set(SRC_HPP
  src/LayerPhaseTestProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
