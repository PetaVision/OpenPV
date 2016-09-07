set(SRC_CPP
  ${TESTS_SHARED_DIR}/Example.cpp
  src/test_mirror_BCs.cpp
)

set(SRC_HPP
  ${TESTS_SHARED_DIR}/Example.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
