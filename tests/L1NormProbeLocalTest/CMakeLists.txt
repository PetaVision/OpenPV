set(SRC_CPP
  src/L1NormProbeLocalTest.cpp
  ${TESTS_SHARED_DIR}/CheckValue.hpp
)

pv_add_test(NO_PARAMS NO_MPI SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
