set(SRC_CPP
  src/PointProbeBatchTest.cpp
  src/TestPointProbe.cpp
)

set(SRC_HPP
  src/TestPointProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
