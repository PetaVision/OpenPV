set(SRC_CPP
  src/StatsProbeLocalTest.cpp
  ${TESTS_SHARED_DIR}/CheckValue.hpp
)

pv_add_test(PARAMS NO_MPI SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
