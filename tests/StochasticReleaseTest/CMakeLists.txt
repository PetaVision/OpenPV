set(SRC_CPP
  src/main.cpp
  src/StochasticReleaseTestProbe.cpp
)

set(SRC_HPP
  src/StochasticReleaseTestProbe.hpp
)

pv_add_test(PARAMS StochasticReleaseTestPre StochasticReleaseTestPost SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
