set(SRC_CPP
  src/HyPerConnCheckpointerTest.cpp
  src/HyPerConnCheckpointerTestProbe.cpp
)

set(SRC_HPP
  src/HyPerConnCheckpointerTestProbe.hpp
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
