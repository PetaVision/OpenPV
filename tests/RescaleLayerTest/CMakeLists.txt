set(SRC_CPP
  src/RescaleLayerTest.cpp
  src/RescaleLayerTestProbe.cpp
)

set(SRC_HPP
  src/RescaleLayerTestProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
