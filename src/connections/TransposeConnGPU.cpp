#include <cmath>
#include <algorithm>
#include "TransposeConnGPU.hpp"
#include "../arch/cuda/CudaUtility.hpp"

namespace GPULCA {

TransposeConnGPU::TransposeConnGPU() {}
	
TransposeConnGPU::TransposeConnGPU(const char *name, PV::HyPerCol *hc)
    : HyPerConnGPU(name, hc) {}

TransposeConnGPU::~TransposeConnGPU() {
  cudnnStatusDestructorCheck(
      cudnnDestroyFilterDescriptor(cudnnFilterDescriptor),
      "destroy filter descriptor");

  cudnnStatusDestructorCheck(
      cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor),
      "destroy convolution descriptor");
}

int TransposeConnGPU::communicateInitInfo() {
  int status = PV_SUCCESS;
  PV::BaseConnection *originalConnBase =
      parent->getConnFromName(this->originalConnName);
  if (originalConnBase == NULL) {
    if (parent->columnId() == 0) {
      fprintf(stderr,
              "%s \"%s\" error: originalConnName \"%s\" does not refer to any "
              "connection in the column.\n",
              this->getKeyword(), name, this->originalConnName);
    }
    MPI_Barrier(parent->getCommunicator()->communicator());
    exit(EXIT_FAILURE);
  }
  this->originalConn = dynamic_cast<HyPerConnGPU *>(originalConnBase);
  if (originalConn == NULL) {
    if (parent->columnId() == 0) {
      fprintf(stderr,
              "TransposeConn \"%s\" error: originalConnName \"%s\" is not an "
              "existing connection.\n",
              name, originalConnName);
      status = PV_FAILURE;
    }
  }
  if (status != PV_SUCCESS) return status;

  if (!originalConn->getInitInfoCommunicatedFlag()) {
    if (parent->columnId() == 0) {
      const char *connectiontype = this->getKeyword();
      printf(
          "%s \"%s\" must wait until original connection \"%s\" has finished "
          "its communicateInitInfo stage.\n",
          connectiontype, name, originalConn->getName());
    }
    return PV_POSTPONE;
  }

  return status;
}

int TransposeConnGPU::allocateDataStructures() {
  try {
    if (getOriginalConn()->getIsWeightSparseFlag()) {
      cerr << "TransposeConnGPU doesn't support sparse weight right now.\n";
      return PV_FAILURE;
    } else {
      int k = getOriginalConn()->getWT().front().dense.getN(),
          c = getOriginalConn()->getWT().front().dense.getC(),
          h = getOriginalConn()->getWT().front().dense.getH(),
          w = getOriginalConn()->getWT().front().dense.getW();

      MatrixInfo wtParams = {
          .n = c, .height = h, .width = w, .channel = k, .layout = NCHW};

      auto initFunc =
          [&](PVCudaWrapper<pvwdata_t> &w) { w.dense.init(wtParams); };
			getWT().resize(numAxonalArborLists);
      std::for_each(getWT().begin(), getWT().end(), initFunc);
      map.init(wtParams);

      cudnnTensorDescriptorPreP =
          &(getOriginalConn()->getPostTensorDescriptor());
      cudnnTensorDescriptorPostP =
          &(getOriginalConn()->getPreTensorDescriptor());

      cudnnStatusCheck(cudnnCreateFilterDescriptor(&cudnnFilterDescriptor),
                       "create filter descriptor");
      cudnnStatusCheck(
          cudnnSetFilter4dDescriptor(
              cudnnFilterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
              getWT().front().dense.getN(), getWT().front().dense.getC(),
              getWT().front().dense.getH(), getWT().front().dense.getW()),
          "set 4D filter");

      const PVLayerLoc *
          preLoc = getOriginalConn()->postSynapticLayer()->getLayerLoc(),
         *postLoc = getOriginalConn()->preSynapticLayer()->getLayerLoc();
      PV::HyPerLayer *pre = getOriginalConn()->postSynapticLayer(),
                     *post = getOriginalConn()->preSynapticLayer();
      int xStride = (post->getCLayer()->xScale - pre->getCLayer()->xScale) * 2,
          yStride = (post->getCLayer()->yScale - pre->getCLayer()->yScale) * 2;
      int pad_h, pad_w, out_h, out_w;
      if (preLoc->nx % xStride != 0 || preLoc->ny % yStride != 0)
        throw logic_error("The size of image is: (" + to_string(preLoc->nx) +
                          "," + to_string(preLoc->ny) + ")\nThe stride is: (" +
                          to_string(xStride) + "," + to_string(yStride) + ").");
      out_h = preLoc->ny / yStride;
      out_w = preLoc->nx / xStride;
      pad_h = (out_h - 1) * yStride + getWT().front().dense.getH() - preLoc->ny;
      pad_w = (out_w - 1) * xStride + getWT().front().dense.getW() - preLoc->nx;

      if (pad_h % 2 != 0 || pad_w % 2 != 0)
        throw logic_error("The pad size is not integer: (" +
                          to_string(pad_h / 2.0) + "," +
                          to_string(pad_w / 2.0) + ").");
      pad_h /= 2;
      pad_w /= 2;
      cudnnStatusCheck(
          cudnnCreateConvolutionDescriptor(&cudnnConvolutionDescriptor),
          "create convolution descriptor");
      cudnnStatusCheck(cudnnSetConvolution2dDescriptor(
                           cudnnConvolutionDescriptor, pad_h, pad_w, xStride,
                           yStride, 1.0, 1.0, CUDNN_CONVOLUTION),
                       "set 2D convolution descriptor");
    }
  } catch (exception &e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
  return PV_SUCCESS;
}

int TransposeConnGPU::deliver() {
  int channelNum = getChannel();
  int preLayerSize = preSynapticLayer()->getCLayer()->numNeurons;
  try {
    if (getOriginalConn()->getIsWeightSparseFlag()) {
      cerr << "TransposeConnGPU doesn't support sparse weight right now.\n";
      return PV_FAILURE;
    } else {
      pvdata_t *preDeviceData =
                   (dynamic_cast<ANNLayerGPU *>(preSynapticLayer()))
                       ->getActivity()
                       .dense.getDeviceData(),
               *postDeviceData =
                   (dynamic_cast<ANNLayerGPU *>(postSynapticLayer()))
                       ->getGSyn()
                       .at(channelNum)
                       .dense.getDeviceData();

      /* convolution  */
      int alpha = 1, beta = 1;
      auto convolveFunc = [&, this](PVCudaWrapper<pvwdata_t> &w) {
        cudnnStatus_t cudnnStatus = cudnnConvolutionForward(
            getOriginalConn()->getCudnnHandle(), &alpha,
            *cudnnTensorDescriptorPreP, preDeviceData, cudnnFilterDescriptor,
            w.dense.getDeviceData(), cudnnConvolutionDescriptor, algoFwd,
            workspaceForward.getDeviceData(), workspaceSizeForward, &beta,
            *cudnnTensorDescriptorPostP, postDeviceData);
        cudnnStatusCheck(cudnnStatus, "convolution");
      };

      std::for_each(getWT().begin(), getWT().end(), convolveFunc);
    }
  } catch (exception &e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
  return PV_SUCCESS;
}

int TransposeConnGPU::findCudnnAlgo() {
  int n, c, h, w;
  try {
    const PVLayerLoc *postLoc = postSynapticLayer()->getLayerLoc();
    cudnnStatusCheck(cudnnGetConvolution2dForwardOutputDim(
                         cudnnConvolutionDescriptor, *cudnnTensorDescriptorPreP,
                         cudnnFilterDescriptor, &n, &c, &h, &w),
                     "cudnnGetConvolution2dForwardOutputDim");

    if (c != postLoc->nf || h != postLoc->ny || w != postLoc->nx) {
      cout << ("Convolution result dimension mismatched.\n" + to_string(c) +
               " " + to_string(h) + " " + to_string(w) + " vs. " +
               to_string(postLoc->nf) + " " + to_string(postLoc->ny) + " " +
               to_string(postLoc->nx) + " ") << endl;
      return PV_FAILURE;
    }

    int m = 8;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> p =
        std::vector<cudnnConvolutionFwdAlgoPerf_t>(m);
    cudnnStatusCheck(
        cudnnFindConvolutionForwardAlgorithm(
            getOriginalConn()->getCudnnHandle(), *cudnnTensorDescriptorPreP,
            cudnnFilterDescriptor, cudnnConvolutionDescriptor,
            *cudnnTensorDescriptorPostP, m, &n, p.data()),
        "cudnnFindConvolutionForwardAlgorithm");

    cudnnStatusCheck(
        cudnnGetConvolutionForwardAlgorithm(
            getOriginalConn()->getCudnnHandle(), *cudnnTensorDescriptorPreP,
            cudnnFilterDescriptor, cudnnConvolutionDescriptor,
            *cudnnTensorDescriptorPostP, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0, &algoFwd),
        "cudnnGetConvolutionForwardAlgorithm");

    cudnnStatusCheck(
        cudnnGetConvolutionForwardWorkspaceSize(
            getOriginalConn()->getCudnnHandle(), *cudnnTensorDescriptorPreP,
            cudnnFilterDescriptor, cudnnConvolutionDescriptor,
            *cudnnTensorDescriptorPostP, algoFwd, &workspaceSizeForward),
        "cudnnGetConvolutionForwardWorkspaceSize");

    workspaceForward.resize(workspaceSizeForward);
  } catch (exception &e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

}  // GPULCA
