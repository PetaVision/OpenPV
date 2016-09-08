set(SRC_CPP
  src/BinningTestLayer.cpp
  src/main.cpp
)

set(SRC_HPP
  src/BinningTestLayer.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
