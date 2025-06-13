set(SRC_CPP
  src/main.cpp
  src/ResetStateOnTriggerTestProbe.cpp
  src/ResetStateOnTriggerTestProbeLocal.cpp
  src/ResetStateOnTriggerTestProbeOutputter.cpp
)

set(SRC_HPP
  src/ResetStateOnTriggerTestProbe.hpp
  src/ResetStateOnTriggerTestProbeOutputter.hpp
)

set(TEST_PARAMS "ResetStateOnTriggerTest" "ResetStateOnTriggerBroadcast")

pv_add_test(PARAMS ${TEST_PARAMS} SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
