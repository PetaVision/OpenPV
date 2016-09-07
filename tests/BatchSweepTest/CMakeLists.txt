set(SRC_CPP
  src/BatchSweepTestProbe.cpp
  src/main.cpp
)

set(SRC_HPP
  src/BatchSweepTestProbe.hpp
)

pv_add_test(MIN_MPI_COPIES 4 MPI_ONLY FLAGS "-batchwidth 4" SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
