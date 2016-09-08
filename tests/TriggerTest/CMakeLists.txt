set(SRC_CPP
  src/main.cpp
  src/TriggerTestConn.cpp
  src/TriggerTestLayer.cpp
  src/TriggerTestLayerProbe.cpp
)

set(SRC_HPP
  src/TriggerTestConn.hpp
  src/TriggerTestLayer.hpp
  src/TriggerTestLayerProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
