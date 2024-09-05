set(SRC_CPP
   src/ConvolutionTest.cpp
)
if (PV_USE_CUDA)
  pv_add_test(PARAMS ConvolutionTest ConvolutionGPUTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
else()
  pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
endif()
