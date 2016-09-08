set(SRC_CPP
  src/main.cpp
)

if (PV_USE_CUDA)
  if (PV_USE_CUDNN)
    pv_add_test(PARAMS ForwardPoolingTest TransposePoolingNonoverlapping SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
  endif (PV_USE_CUDNN)
endif (PV_USE_CUDA)
