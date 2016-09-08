set(SRC_CPP
  src/NormalizeSystemTest.cpp
)

pv_add_test(NO_MPI SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
