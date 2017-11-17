set(SRC_CPP
  src/InitGaborWeights.cpp
  src/InitWeightTestProbe.cpp
  src/NonsharedConnDebugInitWeights.cpp
  src/SharedConnDebugInitWeights.cpp
  src/pv.cpp
)

set(SRC_HPP
  src/InitGaborWeights.hpp
  src/InitWeightTestProbe.hpp
  src/NonsharedConnDebugInitWeights.hpp
  src/SharedConnDebugInitWeights.hpp
)

pv_add_test(PARAMS test_initweights SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
