#include <assert.h>
#include <string.h>
#include "IdentConnGPU.hpp"
#include "../arch/cuda/CudaUtility.hpp"

namespace GPULCA {

IdentConnGPU::IdentConnGPU() { initialize_base(); }

IdentConnGPU::IdentConnGPU(const char *name, PV::HyPerCol *hc)
    : HyPerConnGPU(name, hc) {}

IdentConnGPU::~IdentConnGPU() {}

int IdentConnGPU::communicateInitInfo() {
  int status = HyPerConn::communicateInitInfo();
  assert(pre && post);
  const PVLayerLoc *preLoc = pre->getLayerLoc();
  const PVLayerLoc *postLoc = post->getLayerLoc();
  if (preLoc->nx != postLoc->nx || preLoc->ny != postLoc->ny ||
      preLoc->nf != postLoc->nf) {
    if (parent->columnId() == 0) {
      cerr << "IdentConn \"" << name << "\" Error: " << preLayerName << " and "
           << postLayerName
           << " do not have the same dimensions.\n Dims: " << preLoc->nx << "x"
           << preLoc->ny << "x" << preLoc->nf << " vs. " << postLoc->nx << "x"
           << postLoc->ny << "x" << postLoc->nf << endl;
    }
    exit(EXIT_FAILURE);
  }
  parent->parameters()->handleUnnecessaryParameter(
      name, "nfp", nfp);  // nfp is set during call to
                          // HyPerConn::communicateInitInfo, so don't check for
                          // unnecessary int parameter until after that.
  return status;
}

int IdentConnGPU::deliver() {
  try {
    if (pre->getSparseFlag() || post->getSparseFlag()) {
      cerr << "No sparse IdentConnGPU implementation yet." << endl;
      return PV_FAILURE;
    } else {
      int channelNum = getChannel();
      DenseMatrix<pvdata_t> &postDense =
          (dynamic_cast<ANNLayerGPU *>(postSynapticLayer()))
              ->getGSyn()
              .at(channelNum)
              .dense;

      if (getIsPreGPULayerFlag()) {
        DenseMatrix<pvdata_t> &preDense =
            (dynamic_cast<ANNLayerGPU *>(preSynapticLayer()))
                ->getActivity()
                .dense;
        CudaMatrixAdd<float>(preDense.getSize(), 1, preDense.getDeviceData(),
                                1, postDense.getDeviceData());
        cudaStatusCheck("IdentConnGPU copying from pre to post");
      } else {
        int preLayerSize = preSynapticLayer()->getCLayer()->numNeurons;
        pvdata_t alpha = 1, beta = 1;

        PreNHWC->dense.getCudaVector().setDeviceData(
            preLayerSize, preSynapticLayer()->getActivity());

        cudnnStatus_t cudnnStatus = cudnnTransformTensor(
            cudnnHandle, &alpha, cudnnTensorDescriptorPreNHWC,
            PreNHWC->dense.getDeviceData(), &beta, cudnnTensorDescriptorPre,
            postDense.getDeviceData());
        cudnnStatusCheck(cudnnStatus, "cudnnTransformTensor");
      }
    }
  } catch (exception &e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
}
int IdentConnGPU::allocateDataStructures() {
  if (!getIsPreGPULayerFlag()) {
    try {
      const PVLayerLoc &preLoc = preSynapticLayer()->getCLayer()->loc;

      MatrixInfo paramsNHWC = {.n = preLoc.nbatch,
                               .height = preLoc.ny,
                               .width = preLoc.nx,
                               .channel = preLoc.nf,
                               .layout = NHWC};
      PreNHWC = new PVCudaWrapper<pvwdata_t>(paramsNHWC);

      /*  cuDnn initialization */
      cudnnStatusCheck(
          cudnnSetTensor4dDescriptor(
              cudnnTensorDescriptorPreNHWC, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
              preLoc.nbatch, preLoc.nf, preLoc.ny, preLoc.nx),
          "set 4D tensor");
      cudnnStatusCheck(
          cudnnSetTensor4dDescriptor(
              cudnnTensorDescriptorPre, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
              preLoc.nbatch, preLoc.nf, preLoc.ny, preLoc.nx),
          "set 4D tensor");

    } catch (exception &e) {
      cerr << e.what() << endl;
      return PV_FAILURE;
    }
  }
}

void IdentConnGPU::initialize_base() {}
}
