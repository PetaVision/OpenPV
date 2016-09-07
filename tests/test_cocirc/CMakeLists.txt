set(SRC_CPP
  src/Example.cpp
  src/test_cocirc.cpp
)

set(SRC_HPP
  src/Example.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
