set(SRC_CPP
  src/MomentumConnTestProbe.cpp
  src/MomentumTest.cpp
)

set(SRC_HPP
  src/MomentumConnTestProbe.hpp
)

pv_add_test(PARAMS momentumTest viscosityTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
