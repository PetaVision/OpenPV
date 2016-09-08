set(SRC_CPP
  src/MomentumTestConnProbe.cpp
  src/TestConnProbe.cpp
  src/UpdateFromCloneTest.cpp
)

set(SRC_HPP
  src/MomentumTestConnProbe.hpp
  src/TestConnProbe.hpp
)

pv_add_test(PARAMS updateFromCloneTest momentumUpdateFromCloneTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
