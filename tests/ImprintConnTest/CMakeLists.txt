set(SRC_CPP
  src/ImprintConnTest.cpp
  src/ImprintConnTestInputLayer.cpp
  src/TestLayer.cpp
)

set(SRC_HPP
  src/ImprintConnTestInputLayer.hpp
  src/InputLayer.hpp
  src/TestLayer.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
