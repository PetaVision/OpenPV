set(SRC_CPP
  src/AvgPoolTest.cpp
  src/AvgPoolTestBuffer.cpp
  src/AvgPoolTestInputBuffer.cpp
  src/AvgPoolTestInputLayer.cpp
  src/AvgPoolTestLayer.cpp
  src/GateAvgPoolTestBuffer.cpp
  src/GateAvgPoolTestLayer.cpp
)

set(SRC_HPP
  src/AvgPoolTestBuffer.hpp
  src/AvgPoolTestInputBuffer.hpp
  src/AvgPoolTestInputLayer.hpp
  src/AvgPoolTestLayer.hpp
  src/GateAvgPoolTestBuffer.hpp
  src/GateAvgPoolTestLayer.hpp
)

pv_add_test(PARAMS avgpooltest unpoolTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
