set(SRC_CPP
  src/main.cpp
)

if(${PV_USE_MPI})
  pv_add_test(MIN_MPI_COPIES 10 MAX_MPI_COPIES 10 NO_PARAMS MPI_ONLY SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
endif()
