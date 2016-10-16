#ifndef _CUDAUPDATEWEIGHTS_HPP_
#define _CUDAUPDATEWEIGHTS_HPP_

#include "arch/cuda/CudaKernel.hpp"
#include "include/PVLayerLoc.h"
#include <cudnn.h>

class CUDAUpdateWeights : public CudaKernel {
 public:
  CUDAUpdateWeights();
  vitrual ~CUDAUpdateWeights();

 private:
	cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t cudnnTensorDescriptorPreNHWC,
      cudnnTensorDescriptorPre, cudnnTensorDescriptorPost;
  cudnnFilterDescriptor_t cudnnFilterDescriptor;
  cudnnConvolutionDescriptor_t cudnnConvolutionDescriptor;
  cudnnConvolutionFwdAlgo_t algoFwd;
};

#endif
