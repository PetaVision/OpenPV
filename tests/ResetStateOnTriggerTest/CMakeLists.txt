pv_add_test(PARAMS ResetStateOnTriggerTest)

if (PV_USE_CUDA OR PV_USE_OPENCL)
   pv_add_test(ResetStateOnTriggerGPU)
endif()
