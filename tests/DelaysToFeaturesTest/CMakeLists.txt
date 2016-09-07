set(SRC_CPP
  src/DelayTestProbe.cpp
  src/pv.cpp
)

set(SRC_HPP
  src/DelayTestProbe.hpp
)

pv_add_test(PARAMS test_delays SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
