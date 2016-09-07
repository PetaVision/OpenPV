set(SRC_CPP
  src/main.cpp
  src/ParameterSweepTestProbe.cpp
)

set(SRC_HPP
  src/ParameterSweepTestProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
