set(SRC_CPP
  src/ImageOffsetTest.cpp
  src/ImageOffsetTestActivityBuffer.cpp
  src/ImageOffsetTestLayer.cpp
  src/ImagePvpOffsetTestActivityBuffer.cpp
  src/ImagePvpOffsetTestLayer.cpp
)

set(SRC_HPP
  src/ImageOffsetTestActivityBuffer.hpp
  src/ImageOffsetTestLayer.hpp
  src/ImagePvpOffsetTestActivityBuffer.hpp
  src/ImagePvpOffsetTestLayer.hpp
)

pv_add_test(
  PARAMS ImageOffsetTest ImagePvpOffsetTest AlternateAnchorStrings
  SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H}
)
