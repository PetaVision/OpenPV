set(SRC_CPP
  src/TestNotAlwaysAllZerosProbe.cpp
  src/WriteActivitySparseTest.cpp
)

set(SRC_HPP
  src/TestNotAlwaysAllZerosProbe.hpp
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
