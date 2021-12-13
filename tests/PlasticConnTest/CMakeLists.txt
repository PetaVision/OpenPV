set(SRC_CPP
  src/main.cpp
  src/PlasticConnTestActivityBuffer.cpp
  src/PlasticConnTestLayer.cpp
  src/PlasticTestConn.cpp
  src/PlasticTestUpdater.cpp
)

set(SRC_HPP
  src/PlasticConnTestActivityBuffer.hpp
  src/PlasticConnTestLayer.hpp
  src/PlasticTestConn.hpp
  src/PlasticTestUpdater.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
