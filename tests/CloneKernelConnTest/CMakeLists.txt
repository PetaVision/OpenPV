set(SRC_CPP
  src/CloneKernelConnTest.cpp
  src/CloneKernelConnTestProbe.cpp
)

set(SRC_HPP
  src/CloneKernelConnTestProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
