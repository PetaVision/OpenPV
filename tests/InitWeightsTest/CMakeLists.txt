set(SRC_CPP
  src/HyPerConnDebugInitWeights.cpp
  src/InitGaborWeights.cpp
  src/InitGaborWeightsParams.cpp
  src/InitWeightTestProbe.cpp
  src/KernelConnDebugInitWeights.cpp
  src/pv.cpp
)

set(SRC_HPP
  src/HyPerConnDebugInitWeights.hpp
  src/InitGaborWeights.hpp
  src/InitGaborWeightsParams.hpp
  src/InitWeightTestProbe.hpp
  src/KernelConnDebugInitWeights.hpp
)

pv_add_test(PARAMS test_initweights SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
