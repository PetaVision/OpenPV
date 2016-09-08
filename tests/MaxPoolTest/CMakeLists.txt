set(SRC_CPP
  src/GateMaxPoolTestLayer.cpp
  src/MaxPoolTest.cpp
  src/MaxPoolTestLayer.cpp
)

set(SRC_HPP
  src/GateMaxPoolTestLayer.hpp
  src/MaxPoolTestLayer.hpp
)

pv_add_test(PARAMS maxpooltest gateTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
