set(SRC_CPP
  src/PoolingConnCheckpointerTest.cpp
  src/PoolingConnCheckpointerTestProbe.cpp
  src/CorrectState.cpp
)

set(SRC_HPP
  src/PoolingConnCheckpointerTestProbe.hpp
  src/CorrectState.hpp
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
