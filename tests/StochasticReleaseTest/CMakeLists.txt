set(SRC_CPP
  src/main.cpp
  src/StochasticReleaseTestProbe.cpp
  src/StochasticReleaseTestProbeOutputter.cpp
)

set(SRC_HPP
  src/StochasticReleaseTestProbe.hpp
  src/StochasticReleaseTestProbeOutputter.hpp
)

pv_add_test(PARAMS StochasticReleaseTestPre StochasticReleaseTestPost SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
