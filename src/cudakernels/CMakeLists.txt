if(PV_USE_CUDA)
  set (PVLibSrcCpp ${PVLibSrcCpp}
     ${SUBDIR}/CudaPoolingDeliverKernel.cpp
     ${SUBDIR}/CudaRecvPost.cpp
     ${SUBDIR}/CudaRecvPre.cpp
     ${SUBDIR}/CudaTransposePoolingDeliverKernel.cpp
     ${SUBDIR}/CudaUpdateANNActivity.cpp
     ${SUBDIR}/CudaUpdateHyPerLCAInternalState.cpp
     ${SUBDIR}/CudaUpdateISTAInternalState.cpp
     ${SUBDIR}/CudaUpdateMomentumLCAInternalState.cpp
  )
  
  set (PVLibSrcHpp ${PVLibSrcHpp}
     ${SUBDIR}/CudaPoolingDeliverKernel.hpp
     ${SUBDIR}/CudaRecvPost.hpp
     ${SUBDIR}/CudaRecvPre.hpp
     ${SUBDIR}/CudaTransposePoolingDeliverKernel.hpp
     ${SUBDIR}/CudaUpdateHyPerLCAInternalState.hpp
     ${SUBDIR}/CudaUpdateISTAInternalState.hpp
     ${SUBDIR}/CudaUpdateMomentumLCAInternalState.hpp
  )
  
  set (PVLibSrcCu ${PVLibSrcCu}
     ${SUBDIR}/CudaRecvPre.cu
     ${SUBDIR}/CudaUpdateANNActivity.cu
     ${SUBDIR}/CudaUpdateHyPerLCAInternalState.cu
     ${SUBDIR}/CudaUpdateISTAInternalState.cu
     ${SUBDIR}/CudaUpdateMomentumLCAInternalState.cu
  )
endif(PV_USE_CUDA)
