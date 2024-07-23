set(SRC_CPP
   src/HyPerLCALayerTest.cpp
)
set(HyPerLCALayerGPUTests)
if (PV_USE_CUDA)
  set(HyPerLCALayerGPUTests "BasicRun_GPU")
endif()
pv_add_test(
  PARAMS BasicRun ${HyPerLCALayerGPUTests}
  SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H}
)
