set(SRC_CPP
  src/main.cpp
  src/ResetStateOnTriggerTestProbe.cpp
)

set(SRC_HPP
  src/ResetStateOnTriggerTestProbe.hpp
)

set(TEST_PARAMS "ResetStateOnTriggerTest")

if (PV_USE_CUDA)
  set(TEST_PARAMS "${TEST_PARAMS} ResetStateOnTriggerTestGPU")
endif()

pv_add_test(PARAMS ${testParams} SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})

