set(SRC_CPP
  src/GPUSystemTest.cpp
  src/GPUSystemTestProbe.cpp
  src/identicalBatchProbe.cpp
)

set(SRC_HPP
  src/GPUSystemTestProbe.hpp
  src/identicalBatchProbe.hpp
)

if (PV_USE_CUDA)
  pv_add_test(PARAMS
    preTest
    postTest
    postTestNoTranspose
    postTestOneToMany
    postTest_linked
#   postTestGpuGroup
    HyPerLCAGpuTest
    postRecvBatch
    preRecvBatch
    diffMarginPostTest
    oneToTwo
    oneToTwoTranspose
    twoToOneAccumulation
    SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
endif()
