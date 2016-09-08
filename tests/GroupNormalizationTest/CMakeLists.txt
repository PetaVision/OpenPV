set(SRC_CPP
  src/AllConstantValueProbe.cpp
  src/main.cpp
)

set(SRC_HPP
  src/AllConstantValueProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
