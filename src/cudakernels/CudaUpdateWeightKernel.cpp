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
                                     PVLayerLoc const* _postLoc,
                                     const PVHalo* _preHalo,
                                     const PVHalo* _postHalo, int _numBatch,
                                     CudaBuffer* _errorBuffer,
                                     CudaBuffer* _activityBuffer,
                                     CudaBuffer* _weightBuffer) {
  preLoc = _preLoc;
  postLoc = _postLoc;
  preHalo = _preHalo;
  postHalo = _postHalo;
  numBatch = _numBatch;
  errorBuffer = _errorBuffer;
  activityBuffer = _activityBuffer;
  weightBuffer = _weightBuffer;

  int preToPostScaleX, preToPostScaleY;
  int strideX, strideY;
  int manyScaleX, manyScaleY;
  float fmanyScale;
  preToPostScaleX = preLoc->nx / ((pvdata_t)postLoc->nx);
  preToPostScaleY = preLoc->ny / ((pvdata_t)postLos->ny);

  if (preToPostScaleX < 1) {
    fmanyScale = (float)1 / params.preToPostScaleX;
    manyScaleX = fmanyScale;
    manyScaleY = fmanyScale;
    strideX = 1;
    strideY = 1;
  } else {
    manyScaleX = 1;
    manyScaleY = 1;
    strideX = preToPostScaleX;
    strideY = preToPostScaleY;
  }

  cudnnHandleError(
      cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPre, CUDNN_TENSOR_NCHW,
                                 CUDNN_DATA_FLOAT, numBatch, preLoc->nf,
                                 preLoc->ny, preLoc->nx),
      "set 4D tensor");

  cudnnHandleError(
      cudnnSetTensor4dDescriptor(cudnnTensorDescriptorPost, CUDNN_TENSOR_NCHW,
                                 CUDNN_DATA_FLOAT, 1, postLoc->nf, postLoc->ny,
                                 postLoc->nx),
      "set 4D tensor");

  cudnnHandleError(cudnnStatusCheck(
      cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor, 0, 0, xStride,
                                      yStride, 1.0, 1.0, CUDNN_CONVOLUTION),
      "set 2D convolution descriptor"));

  findCudnnAlgo();
}
}
