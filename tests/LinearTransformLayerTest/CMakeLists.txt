set(SRC_CPP
  src/LinearTransformLayerTest.cpp
)

pv_add_test(
  PARAMS RotateLayerTest ScaleXLayerTest ScaleYLayerTest
  SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H}
)
