if(PV_USE_CUDA)
  set (PVLibSrcCpp ${PVLibSrcCpp}
     ${SUBDIR}/CudaPoolingDeliverKernel.cpp
     ${SUBDIR}/CudaRecvPost.cpp
     ${SUBDIR}/CudaRecvPre.cpp
     ${SUBDIR}/CudaTransposePoolingDeliverKernel.cpp
  )
  
  set (PVLibSrcHpp ${PVLibSrcHpp}
     ${SUBDIR}/CudaPoolingDeliverKernel.hpp
     ${SUBDIR}/CudaRecvPost.hpp
     ${SUBDIR}/CudaRecvPre.hpp
     ${SUBDIR}/CudaTransposePoolingDeliverKernel.hpp
  )
  
  set (PVLibSrcCu ${PVLibSrcCu}
     ${SUBDIR}/CudaRecvPre.cu
  )
endif(PV_USE_CUDA)
