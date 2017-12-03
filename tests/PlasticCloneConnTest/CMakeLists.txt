set(SRC_CPP
  src/PlasticCloneConnTest.cpp
  src/WeightComparisonProbe.cpp
)

set(SRC_HPP
  src/WeightComparisonProbe.hpp
)

pv_add_test(PARAMS PlasticCloneConnTest MomentumPlasticCloneConnTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
