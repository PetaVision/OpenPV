set(SRC_CPP
  src/GateMaxPoolTestBuffer.cpp
  src/GateMaxPoolTestLayer.cpp
  src/MaxPoolTest.cpp
  src/MaxPoolTestBuffer.cpp
  src/MaxPoolTestLayer.cpp
)

set(SRC_HPP
  src/GateMaxPoolTestBuffer.hpp
  src/GateMaxPoolTestLayer.hpp
  src/MaxPoolTestBuffer.hpp
  src/MaxPoolTestLayer.hpp
)

pv_add_test(PARAMS maxpooltest gateTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
