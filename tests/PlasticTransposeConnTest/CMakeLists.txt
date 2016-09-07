set(SRC_CPP
  src/main.cpp
  src/TestAllZerosProbe.cpp
)

set(SRC_HPP
  src/TestAllZerosProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
