set(SRC_CPP
  src/MovieSystemTest.cpp
)

pv_add_test(PARAMS PvpLayerSystemTest ImageLayerSystemTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
