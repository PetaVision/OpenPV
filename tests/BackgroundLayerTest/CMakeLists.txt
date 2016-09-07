set(SRC_CPP
  src/ComparisonLayer.cpp
  src/main.cpp
)

set(SRC_HPP
  src/ComparisonLayer.hpp
)

pv_add_test(PARAMS repTest norepTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
