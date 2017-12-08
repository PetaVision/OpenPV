set(SRC_CPP
  src/main.cpp
  src/TriggerTestConn.cpp
  src/TriggerTestLayer.cpp
  src/TriggerTestLayerProbe.cpp
  src/TriggerTestUpdater.cpp
)

set(SRC_HPP
  src/TriggerTestConn.hpp
  src/TriggerTestLayer.hpp
  src/TriggerTestLayerProbe.hpp
  src/TriggerTestUpdater.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
