set(SRC_CPP
  src/AlwaysFailsLayer.cpp
  src/DryRunFlagTest.cpp
)

set(SRC_HPP
  src/AlwaysFailsLayer.hpp
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
