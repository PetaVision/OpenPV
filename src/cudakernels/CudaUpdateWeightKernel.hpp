#ifndef _CUDAUPDATEWEIGHTS_HPP_
#define _CUDAUPDATEWEIGHTS_HPP_

#include <cudnn.h>
#include "arch/cuda/CudaKernel.hpp"
#include "arch/cuda/CudaVector.hpp"
#include "include/PVLayerLoc.h"
#include "include/pv_datatypes.h"

namespace PVCuda {

class CudaUpdateWeightKernel : public CudaKernel {
 public:
  CudaUpdateWeightKernel(CudaDevice* inDevice);
  virtual ~CudaUpdateWeightKernel();

  void setArgs(PVLayerLoc const* _preLoc, PVLayerLoc const* _postLoc,
               CudaBuffer* _errorBuffer, CudaBuffer* _activityBuffer,
               CudaBuffer* _weightBuffer);

 protected:
  void findCudnnAlgo();

 private:
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t cudnnTensorDescriptorPre, cudnnTensorDescriptorPost;
  cudnnFilterDescriptor_t cudnnFilterDescriptor;
  cudnnConvolutionDescriptor_t cudnnConvolutionDescriptor;
  cudnnConvolutionFwdAlgo_t algoFwd;

  PVLayerLoc const* preLoc;
  PVLayerLoc const* postLoc;
  CudaBuffer* errorBuffer;
  CudaBuffer* activityBuffer;
  CudaBuffer* weightBuffer;

  CudaVector<pvwdata_t> workspaceForward;
  size_t workspaceSizeForward;
};
}
#endif
