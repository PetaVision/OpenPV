set(SRC_CPP
  src/test_extend_border.cpp
)

pv_add_test(NO_PARAMS NO_MPI SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
