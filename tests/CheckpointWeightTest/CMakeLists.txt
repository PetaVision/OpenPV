set(SRC_CPP
  src/CheckpointWeightTest.cpp
)

pv_add_test(NO_PARAMS CheckpointWeightTestShared CheckpointWeightTestNonshared SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})

