set(SRC_CPP
  src/main.cpp
  ${TESTS_SHARED_DIR}/CPTestInputLayer.cpp
  ${TESTS_SHARED_DIR}/IncrementingWeightUpdater.cpp
  ${TESTS_SHARED_DIR}/VaryingHyPerConn.cpp
)

set(SRC_HPP
  ${TESTS_SHARED_DIR}/CPTestInputLayer.hpp
  ${TESTS_SHARED_DIR}/IncrementingWeightUpdater.hpp
  ${TESTS_SHARED_DIR}/VaryingHyPerConn.hpp
)

set(SRC_H
  ${TESTS_SHARED_DIR}/CPTest_updateStateFunctions.h
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
