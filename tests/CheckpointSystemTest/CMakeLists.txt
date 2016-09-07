set(SRC_CPP
  src/CPTestInputLayer.cpp
  src/main.cpp
  src/VaryingHyPerConn.cpp
)

set(SRC_HPP
  src/CPTestInputLayer.hpp
  src/VaryingHyPerConn.hpp
)

set(SRC_H
  src/CPTest_updateStateFunctions.h
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
