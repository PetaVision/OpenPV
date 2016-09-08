set(SRC_CPP
  src/test_mpi_specifyrowscolumns.cpp
)

pv_add_test(NO_PARAMS MPI_ONLY MIN_MPI_COPIES 6 MAX_MPI_COPIES 6 SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
