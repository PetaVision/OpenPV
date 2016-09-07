set(SRC_CPP
  src/AvgPoolTest.cpp
  src/AvgPoolTestInputLayer.cpp
  src/AvgPoolTestLayer.cpp
  src/GatePoolTestLayer.cpp
)

set(SRC_HPP
  src/AvgPoolTestInputLayer.hpp
  src/AvgPoolTestLayer.hpp
  src/GatePoolTestLayer.hpp
)

pv_add_test(PARAMS avgpooltest unpoolTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
