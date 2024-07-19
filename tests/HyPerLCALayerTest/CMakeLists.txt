set(SRC_CPP
   src/HyPerLCALayerTest.cpp
)

pv_add_test(PARAMS BasicRun SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
