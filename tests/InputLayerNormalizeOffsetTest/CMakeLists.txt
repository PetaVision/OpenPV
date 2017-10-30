set(SRC_CPP
  src/InputLayerNormalizeOffsetTest.cpp
)

pv_add_test(PARAMS translate resizePad resizeCrop SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
