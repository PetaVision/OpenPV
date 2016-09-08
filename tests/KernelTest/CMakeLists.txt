set(SRC_CPP
  src/KernelTestProbe.cpp
  src/pv.cpp
)

set(SRC_HPP
  src/KernelTestProbe.hpp
)

pv_add_test(PARAMS test_kernel test_kernel_normalizepost test_kernel_normalizepost_shrunken SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
