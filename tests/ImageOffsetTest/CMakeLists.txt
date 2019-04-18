set(SRC_CPP
  src/ImageOffsetTest.cpp
  src/ImageOffsetTestLayer.cpp
  src/ImagePvpOffsetTestLayer.cpp
)

set(SRC_HPP
  src/ImageOffsetTestLayer.hpp
  src/ImagePvpOffsetTestLayer.hpp
)

pv_add_test(PARAMS ImageOffsetTest ImagePvpOffsetTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
