set(SRC_CPP
  src/TransposeWeightsTest.cpp
  src/TestNonshared.cpp
  src/TestShared.cpp
  src/UtilityFunctions.cpp
)

set(SRC_HPP
  src/TestNonshared.hpp
  src/TestShared.hpp
  src/UtilityFunctions.hpp
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
