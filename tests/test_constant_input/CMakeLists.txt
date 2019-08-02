set(SRC_CPP
  src/test_constant_input.cpp
  src/TestImage.cpp
  src/TestImageActivityComponent.cpp
  src/TestImageActivityBuffer.cpp
)

set(SRC_HPP
  src/TestImage.hpp
  src/TestImageActivityComponent.hpp
  src/TestImageActivityBuffer.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
