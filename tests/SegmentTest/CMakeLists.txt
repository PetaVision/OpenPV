set(SRC_CPP
  src/AssertZerosProbe.cpp
  src/main.cpp
  src/SegmentifyTest.cpp
  src/SegmentTestLayer.cpp
)

set(SRC_HPP
  src/AssertZerosProbe.hpp
  src/SegmentifyTest.hpp
  src/SegmentTestLayer.hpp
)

pv_add_test(PARAMS centerpoint SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})