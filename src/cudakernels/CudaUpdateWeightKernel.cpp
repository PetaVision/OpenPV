#include <include/pv_common.h>
#include <cudakernels/CudaUpdateWeightKernel.hpp>
#include <string>
#include <utils/PVLog.hpp>
#include <vector>

using namespace std;

namespace PVCuda {

CudaUpdateWeightKernel::CudaUpdateWeightKernel(CudaDevice* inDevice)
    : CudaKernel(inDevice) {
  kernelName = "CudaUpdateWeightKernelKernel";

  cudnnHandleError(cudnnCreateTensorDescriptor(&cudnnTensorDescriptorPre),
                   "create tensor descriptor");

  cudnnHandleError(cudnnCreateTensorDescriptor(&cudnnTensorDescriptorPost),
                   "create tensor descriptor");
  cudnnHandleError(
      cudnnCreateConvolutionDescriptor(&cudnnConvolutionDescriptor),
      "create convolution descriptor");
  cudnnHandleError(cudnnCreateFilterDescriptor(&cudnnFilterDescriptor),
                   "create filter descriptor");
}

CudaUpdateWeightKernel::~CudaUpdateWeightKernel() {
  cudnnHandleError(cudnnDestroyTensorDescriptor(cudnnTensorDescriptorPre),
                   "destroy tensor descriptor");
  cudnnHandleError(cudnnDestroyTensorDescriptor(cudnnTensorDescriptorPost),
                   "destroy tensor descriptor");
  cudnnHandleError(cudnnDestroyFilterDescriptor(cudnnFilterDescriptor),
                   "destroy filter descriptor");
  cudnnHandleError(
      cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor),
      "destroy convolution descriptor");
}

void CudaUpdateWeightKernel::findCudnnAlgo() {
  int n, c, h, w;
  cudnnHandleError(cudnnGetConvolution2dForwardOutputDim(
                       cudnnConvolutionDescriptor, cudnnTensorDescriptorPre,
                       cudnnFilterDescriptor, &n, &c, &h, &w),
                   "cudnnGetConvolution2dForwardOutputDim");

  if (c != postLoc->nf || h != postLoc->ny || w != postLoc->nx) {
    pvError() << ("Convolution result dimension mismatched.\n" + to_string(c) +
                  " " + to_string(h) + " " + to_string(w) + " vs. " +
                  to_string(postLoc->nf) + " " + to_string(postLoc->ny) + " " +
                  to_string(postLoc->nx) + " ")
              << endl;
  }

  int m = 8;
  std::vector<cudnnConvolutionFwdAlgoPerf_t> p =
      std::vector<cudnnConvolutionFwdAlgoPerf_t>(m);
  cudnnHandleError(cudnnFindConvolutionForwardAlgorithm(
                       cudnnHandle, cudnnTensorDescriptorPre,
                       cudnnFilterDescriptor, cudnnConvolutionDescriptor,
                       cudnnTensorDescriptorPost, m, &n, p.data()),
                   "cudnnFindConvolutionForwardAlgorithm");

  cudnnHandleError(
      cudnnGetConvolutionForwardAlgorithm(
          cudnnHandle, cudnnTensorDescriptorPre, cudnnFilterDescriptor,
          cudnnConvolutionDescriptor, cudnnTensorDescriptorPost,
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algoFwd),
      "cudnnGetConvolutionForwardAlgorithm");

  cudnnHandleError(
      cudnnGetConvolutionForwardWorkspaceSize(
          cudnnHandle, cudnnTensorDescriptorPre, cudnnFilterDescriptor,
          cudnnConvolutionDescriptor, cudnnTensorDescriptorPost, algoFwd,
          &workspaceSizeForward),
      "cudnnGetConvolutionForwardWorkspaceSize");

  workspaceForward.resize(workspaceSizeForward);
}

void CudaUpdateWeightKernel::setArgs(PVLayerLoc const* _preLoc,
                                PVLayerLoc const* _postLoc,
                                CudaBuffer* _errorBuffer,
                                CudaBuffer* _activityBuffer,
                                CudaBuffer* _weightBuffer) {
  preLoc = _preLoc;
  postLoc = _postLoc;
  errorBuffer = _errorBuffer;
  activityBuffer = _activityBuffer;
  weightBuffer = _weightBuffer;

	cudnnHandleError(
        cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPre, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, 1, preLoc->nf, preLoc->ny,
                                   preLoc->nx),
        "set 4D tensor");

	cudnnHandleError(
        cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPost, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, 1, postLoc->nf,
                                   postLoc->ny, postLoc->nx),
        "set 4D tensor");

  findCudnnAlgo();

	
}
}
