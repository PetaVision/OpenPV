set(SRC_CPP
  src/MPITestActivityBuffer.cpp
  src/MPITestLayer.cpp
  src/MPITestProbe.cpp
  src/MPITestProbeOutputter.cpp
  src/pv.cpp
)

set(SRC_HPP
  src/MPITestActivityBuffer.hpp
  src/MPITestLayer.hpp
  src/MPITestProbe.hpp
  src/MPITestProbeOutputter.hpp
)

pv_add_test(PARAMS MPI_test MPI_ONLY SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
