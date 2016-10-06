set(SRC_CPP
  src/main.cpp
  src/FailBeforeExpectedStartTimeLayer.cpp
)

set(SRC_HPP
  src/copyOutput.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
