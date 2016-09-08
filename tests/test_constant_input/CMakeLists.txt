set(SRC_CPP
  src/test_constant_input.cpp
  src/TestImage.cpp
)

set(SRC_HPP
  src/TestImage.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
