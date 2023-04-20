set(SRC_CPP
   src/TargetLayerComponentTest.cpp
)

pv_add_test(PARAMS NO_MPI SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
