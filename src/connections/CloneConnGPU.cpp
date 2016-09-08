#include "../io/PVParams.hpp"
#include "CloneConnGPU.hpp"

namespace GPULCA {

CloneConnGPU::CloneConnGPU() {}

CloneConnGPU::CloneConnGPU(const char* name, PV::HyPerCol* hc)
    : HyPerConnGPU(name, hc) {}

CloneConnGPU::~CloneConnGPU() {}

int CloneConnGPU::communicateInitInfo() {
  // Need to set originalConn before calling HyPerConn::communicate, since
  // HyPerConn::communicate calls setPatchSize, which needs originalConn.
  BaseConnection* originalConnBase = parent->getConnFromName(originalConnName);
  if (originalConnBase == NULL) {
    if (parent->columnId() == 0) {
      pvErrorNoExit().printf(
          "%s: originalConnName \"%s\" is not a connection in the column.\n",
          getDescription_c(), originalConnName);
    }
    MPI_Barrier(parent->getCommunicator()->communicator());
    exit(EXIT_FAILURE);
  }
  originalConn = dynamic_cast<HyPerConnGPU*>(originalConnBase);
  if (originalConn == NULL) {
    if (parent->columnId() == 0) {
      pvErrorNoExit().printf(
          "%s: originalConnName \"%s\" is not a HyPerConn or HyPerConn-derived "
          "class.\n",
          getDescription_c(), originalConnName);
    }
  }
  if (!originalConn->getInitInfoCommunicatedFlag()) {
    if (parent->columnId() == 0) {
      pvInfo().printf(
          "%s must wait until original connection \"%s\" has finished its "
          "communicateInitInfo stage.\n",
          getDescription_c(), originalConn->getName());
    }
    return PV_POSTPONE;
  }

  // Copy some parameters from originalConn.  Check if parameters exist is
  // the clone's param group, and issue a warning (if the param has the right
  // value) or an error (if it has the wrong value).
  int status = cloneParameters();

  status = HyPerConn::communicateInitInfo();
  if (status != PV_SUCCESS) return status;

  // Presynaptic layers of the CloneConn and its original conn must have the
  // same size, or the patches won't line up with each other.
  const PVLayerLoc* preLoc = pre->getLayerLoc();
  const PVLayerLoc* origPreLoc =
      originalConn->preSynapticLayer()->getLayerLoc();

  if (preLoc->nx != origPreLoc->nx || preLoc->ny != origPreLoc->ny ||
      preLoc->nf != origPreLoc->nf) {
    if (parent->getCommunicator()->commRank() == 0) {
      pvErrorNoExit(errorMessage);
      errorMessage.printf(
          "%s: CloneConn and originalConn \"%s\" must have presynaptic layers "
          "with the same nx,ny,nf.\n",
          getDescription_c(), parent->columnId(), originalConn->getName());
      errorMessage.printf(
          "{nx=%d, ny=%d, nf=%d} versus {nx=%d, ny=%d, nf=%d}\n", preLoc->nx,
          preLoc->ny, preLoc->nf, origPreLoc->nx, origPreLoc->ny,
          origPreLoc->nf);
    }
    MPI_Barrier(parent->getCommunicator()->communicator());
    abort();
  }

  // Make sure the original's and the clone's margin widths stay equal
  originalConn->preSynapticLayer()->synchronizeMarginWidth(pre);
  pre->synchronizeMarginWidth(originalConn->preSynapticLayer());

  // Make sure the original's and the clone's margin widths stay equal
  // Only if this layer receives from post for patch to data LUT
  if (getUpdateGSynFromPostPerspective()) {
    originalConn->postSynapticLayer()->synchronizeMarginWidth(post);
    post->synchronizeMarginWidth(originalConn->postSynapticLayer());
  }
  // Redudant read in case it's a clone of a clone

  return status;
}

int CloneConnGPU::allocateDataStructures() {
  int status = HyPerConn::allocateDataStructures();
  try {
    if (getOriginalConn()->getIsWeightSparseFlag()) {
      cerr << "TransposeConnGPU doesn't support sparse weight right now.\n";
      return PV_FAILURE;
    } else {
      const PVLayerLoc* preLoc = preSynapticLayer()->getLayerLoc(),
                        * postLoc = postSynapticLayer()->getLayerLoc();
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
            &((dynamic_cast<ANNLayerGPU*>(preSynapticLayer()))->getActivity());

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
              getOriginalConn()->getW().front().dense.getN(),
              getOriginalConn()->getW().front().dense.getC(),
              getOriginalConn()->getW().front().dense.getH(),
              getOriginalConn()->getW().front().dense.getW()),
          "set 4D filter");

      PV::HyPerLayer* pre = getOriginalConn()->postSynapticLayer(),
                      * post = getOriginalConn()->preSynapticLayer();
      int xStride = (post->getCLayer()->xScale - pre->getCLayer()->xScale) * 2,
          yStride = (post->getCLayer()->yScale - pre->getCLayer()->yScale) * 2;
      int pad_h, pad_w, out_h, out_w;
      if (preLoc->nx % xStride != 0 || preLoc->ny % yStride != 0)
        throw logic_error("The size of image is: (" + to_string(preLoc->nx) +
                          "," + to_string(preLoc->ny) + ")\nThe stride is: (" +
                          to_string(xStride) + "," + to_string(yStride) + ").");
      out_h = preLoc->ny / yStride;
      out_w = preLoc->nx / xStride;
      pad_h = (out_h - 1) * yStride + getOriginalConn()->getW().front().dense.getH() - preLoc->ny;
      pad_w = (out_w - 1) * xStride + getOriginalConn()->getW().front().dense.getW() - preLoc->nx;

      if (pad_h % 2 != 0 || pad_w % 2 != 0)
        throw logic_error("The pad size is not integer: (" +
                          to_string(pad_h / 2.0) + "," +
                          to_string(pad_w / 2.0) + ").");
      pad_h /= 2;
      pad_w /= 2;

      cudnnStatusCheck(cudnnSetConvolution2dDescriptor(
                           cudnnConvolutionDescriptor, pad_h, pad_w, xStride,
                           yStride, 1.0, 1.0, CUDNN_CROSS_CORRELATION),
                       "set 2D convolution descriptor");
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  status = findCudnnAlgo();

  return status;
}

int CloneConnGPU::deliver() {
  int channelNum = getChannel();
  int preLayerSize = preSynapticLayer()->getCLayer()->numNeurons;
  try {
    if (getOriginalConn()->getIsWeightSparseFlag()) {
      cerr << "TransposeConnGPU doesn't support sparse weight right now.\n";
      return PV_FAILURE;
    } else {
      if (!getIsPreGPULayerFlag()) {
        pvdata_t* preHostData = preSynapticLayer()->getActivity();

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

      pvdata_t* postDeviceData =
          (dynamic_cast<ANNLayerGPU*>(postSynapticLayer()))
              ->getGSyn()
              .at(channelNum)
              .dense.getDeviceData();

      /* convolution  */
      int alpha = 1, beta = 1;
      auto convolveFunc = [&, this](PVCudaWrapper<pvwdata_t>& w) {
        cudnnStatus_t cudnnStatus = cudnnConvolutionForward(
            cudnnHandle, &alpha, *cudnnTensorDescriptorPreP,
            Pre->dense.getDeviceData(), cudnnFilterDescriptor,
            w.dense.getDeviceData(), cudnnConvolutionDescriptor, algoFwd,
            workspaceForward.getDeviceData(), workspaceSizeForward, &beta,
            *cudnnTensorDescriptorPostP, postDeviceData);
        cudnnStatusCheck(cudnnStatus, "convolution");
      };

      std::for_each(getOriginalConn()->getW().begin(),
                    getOriginalConn()->getW().end(), convolveFunc);
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

int CloneConnGPU::cloneParameters() {
  // Copy sharedWeights, numAxonalArborLists, shrinkPatches_flag from
  // originalConn

  PV::PVParams* params = parent->parameters();

  sharedWeights = originalConn->usingSharedWeights();
  params->handleUnnecessaryParameter(name, "sharedWeights", sharedWeights);

  numAxonalArborLists = originalConn->numberOfAxonalArborLists();
  params->handleUnnecessaryParameter(name, "numAxonalArbors",
                                     numAxonalArborLists);

  shrinkPatches_flag = originalConn->getShrinkPatches_flag();
  parent->parameters()->handleUnnecessaryParameter(name, "shrinkPatches",
                                                   shrinkPatches_flag);
  return PV_SUCCESS;
}
}
