set(SRC_CPP
  src/MomentumDecayTest.cpp
)

pv_add_test(PARAMS weightL1Decay weightL2Decay weightsElasticNet SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
