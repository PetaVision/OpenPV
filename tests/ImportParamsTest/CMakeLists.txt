set(SRC_CPP
  src/ImportParamsConn.cpp
  src/ImportParamsLayer.cpp
  src/main.cpp
)

set(SRC_HPP
  src/ImportParamsConn.hpp
  src/ImportParamsLayer.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
