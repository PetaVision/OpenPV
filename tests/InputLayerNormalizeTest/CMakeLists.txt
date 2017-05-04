set(SRC_CPP
  src/CheckStatsProbe.cpp
  src/InputLayerNormalizeTest.cpp
)

set(SRC_HPP
  src/CheckStatsProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
