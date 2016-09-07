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

pv_add_test(MAX_MPI_COPIES 8 NO_PARAMS MPI_ONLY SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})