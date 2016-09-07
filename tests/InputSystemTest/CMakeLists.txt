set(SRC_CPP
  src/MovieSystemTest.cpp
  src/TestAllZerosProbe.cpp
)

set(SRC_HPP
  src/TestAllZerosProbe.hpp
)

pv_add_test(PARAMS PvpLayerSystemTest ImageLayerSystemTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
