if(PV_USE_CUDA)
  set (PVLibSrcHpp ${PVLibSrcHpp}
     ${SUBDIR}/CudaPoolingDeliverKernel.hpp
     ${SUBDIR}/CudaRecvPost.hpp
     ${SUBDIR}/CudaRecvPre.hpp
     ${SUBDIR}/CudaTransposePoolingDeliverKernel.hpp
     ${SUBDIR}/CudaUpdateStateFunctions.hpp
  )
  
  set (PVLibSrcCpp ${PVLibSrcCpp}
     ${SUBDIR}/CudaPoolingDeliverKernel.cpp
     ${SUBDIR}/CudaRecvPost.cpp
     ${SUBDIR}/CudaRecvPre.cpp
     ${SUBDIR}/CudaTransposePoolingDeliverKernel.cpp
     ${SUBDIR}/CudaUpdateStateFunctions.cpp
  )
  
  set (PVLibSrcCu ${PVLibSrcCu}
     ${SUBDIR}/CudaRecvPre.cu
     ${SUBDIR}/CudaUpdateStateFunctions.cu
  )
endif(PV_USE_CUDA)
