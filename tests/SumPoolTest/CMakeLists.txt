set(SRC_CPP
  src/GateSumPoolTestBuffer.cpp
  src/GateSumPoolTestLayer.cpp
  src/SumPoolTest.cpp
  src/SumPoolTestInputBuffer.cpp
  src/SumPoolTestInputLayer.cpp
  src/SumPoolTestBuffer.cpp
  src/SumPoolTestLayer.cpp
)

set(SRC_HPP
  src/GateSumPoolTestBuffer.hpp
  src/GateSumPoolTestLayer.hpp
  src/SumPoolTestInputBuffer.hpp
  src/SumPoolTestInputLayer.hpp
  src/SumPoolTestBuffer.hpp
  src/SumPoolTestLayer.hpp
)

pv_add_test(PARAMS sumpooltest unpoolTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
