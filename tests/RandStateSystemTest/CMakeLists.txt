set(SRC_CPP
  src/main.cpp
  ${TESTS_SHARED_DIR}/ColumnArchive.cpp
)

set(SRC_HPP
  ${TESTS_SHARED_DIR}/ColumnArchive.hpp
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
