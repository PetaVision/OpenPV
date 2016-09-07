set(SRC_CPP
  src/CloneHyPerConnTest.cpp
  src/CloneHyPerConnTestProbe.cpp
)

set(SRC_HPP
  src/CloneHyPerConnTestProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
