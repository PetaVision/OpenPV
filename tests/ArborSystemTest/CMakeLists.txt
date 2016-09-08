set(SRC_CPP
  src/ArborTestForOnesProbe.cpp
  src/ArborTestProbe.cpp
  src/pv.cpp
)

set(SRC_HPP
  src/ArborTestForOnesProbe.hpp
  src/ArborTestProbe.hpp
)

pv_add_test(PARAMS test_arbors SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
