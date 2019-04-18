set(SRC_CPP
  src/main.cpp
  src/PlasticConnTestLayer.cpp
  src/PlasticConnTestProbe.cpp
  src/PlasticTestConn.cpp
  src/PlasticTestUpdater.cpp
)

set(SRC_HPP
  src/PlasticConnTestLayer.hpp
  src/PlasticConnTestProbe.hpp
  src/PlasticTestConn.hpp
  src/PlasticTestUpdater.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
