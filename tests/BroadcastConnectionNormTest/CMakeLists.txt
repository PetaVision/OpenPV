set(SRC_CPP
   src/BroadcastConnectionNormTest.cpp
)

if (PV_USE_MPI)
  pv_add_test(MPI_ONLY MIN_MPI_COPIES 4 MAX_MPI_COPIES 4 SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
endif (PV_USE_MPI)

