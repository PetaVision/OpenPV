set(SRC_CPP
  src/GateSumPoolTestLayer.cpp
  src/SumPoolTest.cpp
  src/SumPoolTestInputLayer.cpp
  src/SumPoolTestLayer.cpp
)

set(SRC_HPP
  src/GateSumPoolTestLayer.hpp
  src/SumPoolTestInputLayer.hpp
  src/SumPoolTestLayer.hpp
)

pv_add_test(PARAMS sumpooltest unpoolTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
