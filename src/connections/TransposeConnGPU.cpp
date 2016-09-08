#include <cmath>
#include <algorithm>
#include "TransposeConnGPU.hpp"
#include "../arch/cuda/CudaUtility.hpp"

namespace GPULCA {

TransposeConnGPU::TransposeConnGPU() {}

TransposeConnGPU::TransposeConnGPU(const char *name, PV::HyPerCol *hc)
    : HyPerConnGPU(name, hc) {}

TransposeConnGPU::~TransposeConnGPU() {}

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
  int status = HyPerConn::allocateDataStructures();
  try {
    if (getOriginalConn()->getIsWeightSparseFlag()) {
      cerr << "TransposeConnGPU doesn't support sparse weight right now.\n";
      return PV_FAILURE;
    } else {
      const PVLayerLoc *preLoc = preSynapticLayer()->getLayerLoc(),
                       *postLoc = postSynapticLayer()->getLayerLoc();
      MatrixInfo preNHWCParams = {.n = 1,
                                  .height = preLoc->ny,
                                  .width = preLoc->nx,
                                  .channel = preLoc->nf,
                                  .layout = NHWC},
                 preParams = {.n = 1,
                              .height = preLoc->ny,
                              .width = preLoc->nx,
                              .channel = preLoc->nf,
                              .layout = NCHW};
      if (getIsPreGPULayerFlag()) {
        PreNHWC = nullptr;
        Pre =
            &((dynamic_cast<ANNLayerGPU *>(preSynapticLayer()))->getActivity());

      } else {
        PreNHWC = new PVCudaWrapper<pvwdata_t>(preNHWCParams);
        Pre = new PVCudaWrapper<pvwdata_t>(preParams);
      }

      cudnnTensorDescriptorPreP =
          &(getOriginalConn()->getPostTensorDescriptor());
      cudnnTensorDescriptorPostP =
          &(getOriginalConn()->getPreTensorDescriptor());

      cudnnStatusCheck(
          cudnnSetFilter4dDescriptor(
              cudnnFilterDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
              getOriginalConn()->getWT().front().dense.getN(),
              getOriginalConn()->getWT().front().dense.getC(),
              getOriginalConn()->getWT().front().dense.getH(),
              getOriginalConn()->getWT().front().dense.getW()),
          "set 4D filter");

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
      pad_h = (out_h - 1) * yStride + getOriginalConn()->getWT().front().dense.getH() - preLoc->ny;
      pad_w = (out_w - 1) * xStride + getOriginalConn()->getWT().front().dense.getW() - preLoc->nx;

      if (pad_h % 2 != 0 || pad_w % 2 != 0)
        throw logic_error("The pad size is not integer: (" +
                          to_string(pad_h / 2.0) + "," +
                          to_string(pad_w / 2.0) + ").");
      pad_h /= 2;
      pad_w /= 2;

      cudnnStatusCheck(cudnnSetConvolution2dDescriptor(
                           cudnnConvolutionDescriptor, pad_h, pad_w, xStride,
                           yStride, 1.0, 1.0, CUDNN_CONVOLUTION),
                       "set 2D convolution descriptor");
    }
  } catch (exception &e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  status = findCudnnAlgo();

  return status;
}

int TransposeConnGPU::deliver() {
  int channelNum = getChannel();
  int preLayerSize = preSynapticLayer()->getCLayer()->numNeurons;
  try {
    if (getOriginalConn()->getIsWeightSparseFlag()) {
      cerr << "TransposeConnGPU doesn't support sparse weight right now.\n";
      return PV_FAILURE;
    } else {
      if (!getIsPreGPULayerFlag()) {
        pvdata_t *preHostData = preSynapticLayer()->getActivity();

        PreNHWC->dense.getCudaVector().setDeviceData(preLayerSize, preHostData);

        /* change pre-layer layout (need a parameter to specify whether
         * pre-layer is a GPULCA layer or not ) */
        pvdata_t alpha = 1, beta = 0;
        cudnnStatus_t cudnnStatus = cudnnTransformTensor(
            cudnnHandle, &alpha, cudnnTensorDescriptorPreNHWC,
            PreNHWC->dense.getDeviceData(), &beta, cudnnTensorDescriptorPre,
            Pre->dense.getDeviceData());
        cudnnStatusCheck(cudnnStatus, "cudnnTransformTensor");
      }

      pvdata_t *postDeviceData =
          (dynamic_cast<ANNLayerGPU *>(postSynapticLayer()))
              ->getGSyn()
              .at(channelNum)
              .dense.getDeviceData();

      /* convolution  */
      int alpha = 1, beta = 1;
      auto convolveFunc = [&, this](PVCudaWrapper<pvwdata_t> &w) {
        cudnnStatus_t cudnnStatus = cudnnConvolutionForward(
            cudnnHandle, &alpha, *cudnnTensorDescriptorPreP, Pre->dense.getDeviceData(),
            cudnnFilterDescriptor, w.dense.getDeviceData(),
            cudnnConvolutionDescriptor, algoFwd,
            workspaceForward.getDeviceData(), workspaceSizeForward, &beta,
            *cudnnTensorDescriptorPostP, postDeviceData);
        cudnnStatusCheck(cudnnStatus, "convolution");
      };

      std::for_each(getOriginalConn()->getWT().begin(),
                    getOriginalConn()->getWT().end(), convolveFunc);
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
    cudnnStatusCheck(cudnnFindConvolutionForwardAlgorithm(
                         cudnnHandle, *cudnnTensorDescriptorPreP,
                         cudnnFilterDescriptor, cudnnConvolutionDescriptor,
                         *cudnnTensorDescriptorPostP, m, &n, p.data()),
                     "cudnnFindConvolutionForwardAlgorithm");

    cudnnStatusCheck(
        cudnnGetConvolutionForwardAlgorithm(
            cudnnHandle, *cudnnTensorDescriptorPreP, cudnnFilterDescriptor,
            cudnnConvolutionDescriptor, *cudnnTensorDescriptorPostP,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algoFwd),
        "cudnnGetConvolutionForwardAlgorithm");

    cudnnStatusCheck(
        cudnnGetConvolutionForwardWorkspaceSize(
            cudnnHandle, *cudnnTensorDescriptorPreP, cudnnFilterDescriptor,
            cudnnConvolutionDescriptor, *cudnnTensorDescriptorPostP, algoFwd,
            &workspaceSizeForward),
        "cudnnGetConvolutionForwardWorkspaceSize");

    workspaceForward.resize(workspaceSizeForward);
  } catch (exception &e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

}  // GPULCA
