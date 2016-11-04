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
  int n, m = 8;
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> p =
      std::vector<cudnnConvolutionBwdFilterAlgoPerf_t>(m);
  cudnnHandleError(
      cudnnFindConvolutionBackwardFilterAlgorithm(
          (cudnnHandle_t)device->getCudnnHandle(), cudnnTensorDescriptorPre,
          cudnnTensorDescriptorPost, cudnnConvolutionDescriptor,
          cudnnFilterDescriptor, m, &n, p.data()),
      "cudnnFindConvolutionBackwardFilterAlgorithm");

  cudnnHandleError(
      cudnnGetConvolutionBackwardFilterAlgorithm(
          (cudnnHandle_t)device->getCudnnHandle(), cudnnTensorDescriptorPre,
          cudnnTensorDescriptorPost, cudnnConvolutionDescriptor,
          cudnnFilterDescriptor, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
          &algoBwd),
      "cudnnGetConvolutionBackwardFilterAlgorithm");

  cudnnHandleError(
      cudnnGetConvolutionBackwardFilterWorkspaceSize(
          (cudnnHandle_t)device->getCudnnHandle(), cudnnTensorDescriptorPre,
          cudnnTensorDescriptorPost, cudnnConvolutionDescriptor,
          cudnnFilterDescriptor, algoBwd, &workspaceSizeBackwardFilter),
      "cudnnGetConvolutionBackwardFilterWorkspaceSize");

  workspaceBackwardFilter.resize(workspaceSizeBackwardFilter);
}

void CudaUpdateWeightKernel::setArgs(PVLayerLoc const* _preLoc,
                                     PVLayerLoc const* _postLoc, int _numBatch,
                                     int nxpPost, int nypPost,
                                     CudaBuffer* _errorBuffer,
                                     CudaBuffer* _activityBuffer,
                                     CudaBuffer* _weightBuffer) {
  preLoc = _preLoc;
  postLoc = _postLoc;
  preHalo = &_preLoc->halo;
  postHalo = &_postLoc->halo;
  numBatch = _numBatch;
  errorBuffer = _errorBuffer;
  activityBuffer = _activityBuffer;
  weightBuffer = _weightBuffer;

  int preToPostScaleX, preToPostScaleY;
  int strideX, strideY;
  int manyScaleX, manyScaleY;
  float fmanyScale;
  preToPostScaleX = preLoc->nx / ((pvdata_t)postLoc->nx);
  preToPostScaleY = preLoc->ny / ((pvdata_t)postLoc->ny);

  // One to many
  if (preToPostScaleX < 1) {
    strideX = 1;
    strideY = 1;

    cudnnHandleError(
        cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPre, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, numBatch, preLoc->nf,
                                   preLoc->ny + nypPost, preLoc->nx + nxpPost),
        "set 4D tensor");

  } else {  // many to one
    strideX = preToPostScaleX;
    strideY = preToPostScaleY;

    cudnnHandleError(
        cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPre, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, numBatch, preLoc->nf,
                                   preLoc->ny + nypPost - preToPostScaleY,
                                   preLoc->nx + nxpPost - preToPostScaleX),
        "set 4D tensor");
  }

  cudnnHandleError(
      cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPost, CUDNN_TENSOR_NCHW,
                                 CUDNN_DATA_FLOAT, numBatch, postLoc->nf,
                                 postLoc->ny, postLoc->nx),
      "set 4D tensor");

  cudnnHandleError(
      cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor, 0, 0, strideY,
                                      strideX, 1.0, 1.0, CUDNN_CONVOLUTION),
      "set 2D convolution descriptor");

  cudnnHandleError(cudnnSetFilter4dDescriptor(
      cudnnFilterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, postLoc->nf,
      preLoc->nf, preLoc->nyp, preLoc->nxp));


	// test dimension setting
	int n, c, h, w;
  cudnnHandleError(cudnnGetConvolution2dForwardOutputDim(
                       cudnnConvolutionDescriptor, cudnnTensorDescriptorPre,
                       cudnnFilterDescriptor, &n, &c, &h, &w),
                   "cudnnGetConvolution2dForwardOutputDim");

  if (c != postLoc->nf || h != postLoc->ny || w != postLoc->nx || n != numBatch) {
    pvError() << ("Convolution result dimension mismatched.\n" + to_string(n) + " " + to_string(c) +
                  " " + to_string(h) + " " + to_string(w) + " vs. " + to_string(numBatch) + " " + 
                  to_string(postLoc->nf) + " " + to_string(postLoc->ny) + " " +
                  to_string(postLoc->nx) + " ")
              << endl;
  }

  findCudnnAlgo();
}
}
