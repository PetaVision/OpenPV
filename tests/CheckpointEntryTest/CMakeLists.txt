set(SRC_CPP
  src/main.cpp
  src/testSeparatedName.cpp
  src/testDataNoBroadcast.cpp
  src/testDataWithBroadcast.cpp
  src/testPvpExtended.cpp
  src/testPvpRestricted.cpp
  src/testPvpBatch.cpp
)

set(SRC_HPP
  src/testSeparatedName.cpp
  src/testDataNoBroadcast.hpp
  src/testDataWithBroadcast.hpp
  src/testPvpExtended.hpp
  src/testPvpRestricted.hpp
  src/testPvpBatch.hpp
)

pv_add_test(NO_PARAMS SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
