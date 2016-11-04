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
               int _numBatch, int nxpPost, int nypPost, CudaBuffer* _errorBuffer,
               CudaBuffer* _activityBuffer, CudaBuffer* _weightBuffer);

 protected:
  void findCudnnAlgo();

 private:
  cudnnTensorDescriptor_t cudnnTensorDescriptorPre, cudnnTensorDescriptorPost;
  cudnnFilterDescriptor_t cudnnFilterDescriptor;
  cudnnConvolutionDescriptor_t cudnnConvolutionDescriptor;
  cudnnConvolutionBwdFilterAlgo_t algoBwd;

  PVLayerLoc const* preLoc;
  PVLayerLoc const* postLoc;
  const PVHalo* preHalo;
  const PVHalo* postHalo;
  int numBatch;
  CudaBuffer* errorBuffer;
  CudaBuffer* activityBuffer;
  CudaBuffer* weightBuffer;

  CudaVector<pvwdata_t> workspaceBackwardFilter;
  size_t workspaceSizeBackwardFilter;
};
}
#endif
