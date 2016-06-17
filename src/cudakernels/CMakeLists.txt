if(PV_USE_CUDA)
  set (PVLibSrcHpp ${PVLibSrcHpp}
     ${SUBDIR}/CudaRecvPost.hpp
     ${SUBDIR}/CudaRecvPre.hpp
     ${SUBDIR}/CudaUpdateStateFunctions.hpp
  )
  
  set (PVLibSrcCpp ${PVLibSrcCpp}
     ${SUBDIR}/CudaRecvPost.cpp
     ${SUBDIR}/CudaRecvPre.cpp
     ${SUBDIR}/CudaUpdateStateFunctions.cpp
  )
  
  set (PVLibSrcCu ${PVLibSrcCu}
     ${SUBDIR}/CudaRecvPost.cu
     ${SUBDIR}/CudaRecvPre.cu
     ${SUBDIR}/CudaUpdateStateFunctions.cu
  )
endif(PV_USE_CUDA)
