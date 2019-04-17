set(SRC_CPP
  src/DatastoreDelayTestBuffer.cpp
  src/DatastoreDelayTestLayer.cpp
  src/DatastoreDelayTestProbe.cpp
  src/main.cpp
)

set(SRC_HPP
  src/DatastoreDelayTestBuffer.hpp
  src/DatastoreDelayTestLayer.hpp
  src/DatastoreDelayTestProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
