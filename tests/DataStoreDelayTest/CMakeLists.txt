set(SRC_CPP
  src/DataStoreDelayTestBuffer.cpp
  src/DataStoreDelayTestLayer.cpp
  src/DataStoreDelayTestProbe.cpp
  src/main.cpp
)

set(SRC_HPP
  src/DataStoreDelayTestBuffer.hpp
  src/DataStoreDelayTestLayer.hpp
  src/DataStoreDelayTestProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
