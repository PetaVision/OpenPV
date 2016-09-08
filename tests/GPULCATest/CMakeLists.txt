set(SRC_CPP
  src/main.cpp
)

if (PV_USE_CUDA)
   pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
endif (PV_USE_CUDA)
