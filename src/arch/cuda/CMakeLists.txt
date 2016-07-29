if(PV_USE_CUDA)
  set (PVLibSrcHpp ${PVLibSrcHpp}
     ${SUBDIR}/cuda_util.hpp
     ${SUBDIR}/CudaBuffer.hpp
     ${SUBDIR}/CudaDevice.hpp
     ${SUBDIR}/CudaKernel.hpp
     ${SUBDIR}/CudaTimer.hpp
		 ${SUBDIR}/CudaVector.hpp
		 ${SUBDIR}/CudaMatrix.hpp
		 ${SUBDIR}/CudaUtility.hpp
  )
  
  set (PVLibSrcCpp ${PVLibSrcCpp}
     ${SUBDIR}/CudaBuffer.cpp
     ${SUBDIR}/CudaDevice.cpp
     ${SUBDIR}/CudaKernel.cpp
     ${SUBDIR}/CudaTimer.cpp
  )
  
  set (PVLibSrcCu ${PVLibSrcCu}
    ${SUBDIR}/CudaBuffer.cu
		${SUBDIR}/CudaUtility.cu
  )
endif(PV_USE_CUDA)
