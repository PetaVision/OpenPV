#include <cublas_v2.h>
#include "LateralInteractionConnGPU.hpp"
#include "../layers/HyPerGPULCALayer.hpp"

namespace GPULCA {

LateralInteractionConnGPU::LateralInteractionConnGPU() {}

LateralInteractionConnGPU::LateralInteractionConnGPU(const char *name,
                                                     PV::HyPerCol *hc)
    : HyPerConnGPU(name, hc) {}

LateralInteractionConnGPU::~LateralInteractionConnGPU() {}

int LateralInteractionConnGPU::communicateInitInfo() {
  int status = PV_SUCCESS;
  BaseConnection *originalConnBase =
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

int LateralInteractionConnGPU::allocateDataStructures() {
  try {
    if (getOriginalConn()->getIsWeightSparseFlag()) {
      cerr << "GPUTransposeConn doesn't support sparse weight right now.\n";
      return PV_FAILURE;
    } else {
      int nf = getOriginalConn()->getWT().front().dense.getN();
      MatrixInfo wwParams = {
          .n = 1, .height = nf, .width = nf, .channel = 1, .layout = NCHW};

      getWT().resize(1);
      auto initFunc =
          [&](PVCudaWrapper<pvwdata_t> &w) { w.dense.init(wwParams); };
      std::for_each(getWT().begin(), getWT().end(), initFunc);
    }
  } catch (exception &e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
  return PV_SUCCESS;
}

int LateralInteractionConnGPU::deliver() {
  int channelNum = getChannel();
  try {
    if (getOriginalConn()->getIsWeightSparseFlag()) {
      cerr << "LateralInteractionConnGPU doesn't support sparse weight right "
           << "now.\n";
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
      int postlda = (dynamic_cast<ANNLayerGPU *>(postSynapticLayer()))
                        ->getGSyn()
                        .at(channelNum)
                        .dense.getDevice2DRows();
      cublasStatus_t cublasStatus;
      cublasOperation_t transa, transb;
      int m, n, k, lda, ldb, ldc;
      float alpha = 1, beta = 0;
      m = getWT().front().dense.getH();
      n = getWT().front().dense.getW();
      lda = getOriginalConn()->getWT().front().dense.getDevice2DRows();
      ldb = getOriginalConn()->getWT().front().dense.getDevice2DRows();
      ldc = getWT().front().dense.getH();

      if (m != n || m != ldc)
        throw logic_error("One of the dimension of WWT should equal to nf.");

      if (lda == ldc) {
        transa = CUBLAS_OP_N;
        transb = CUBLAS_OP_T;
        k = getOriginalConn()->getWT().front().dense.getDevice2DCols();
      } else {
        transa = CUBLAS_OP_T;
        transb = CUBLAS_OP_N;
        k = getOriginalConn()->getWT().front().dense.getDevice2DRows();
      }

      const cusparseHandle_t &cusparseHandle =
          (dynamic_cast<ANNLayerGPU *>(preSynapticLayer()))
              ->getCusparseHandle();
      const cusparseMatDescr_t &cusparseMatDescr =
          (dynamic_cast<ANNLayerGPU *>(preSynapticLayer()))
              ->getCusparseMatDescr();
      SparseMatrix<pvdata_t> &sparseA =
          (dynamic_cast<ANNLayerGPU *>(preSynapticLayer()))
              ->getActivity()
              .sparse;

      auto func = [&, this](PVCudaWrapper<pvwdata_t> &w) {
        cublasStatusCheck(
            cublasSgemm((dynamic_cast<HyPerGPULCALayer *>(preSynapticLayer()))
                            ->getCublasHandle(),
                        transa, transb, m, n, k, &alpha,
                        w.dense.getDeviceData(), lda, w.dense.getDeviceData(),
                        ldb, &beta, getWT().front().dense.getDeviceData(), ldc),
            "computeWWT");

        cusparseStatusCheck(
            cusparseScsrmm(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                sparseA.getDevice2DRows(), getWT().front().dense.getW(),
                sparseA.getDevice2DCols(), sparseA.getNNZ(), &alpha,
                cusparseMatDescr, sparseA.getCooCsrValA(),
                sparseA.getCsrRowIndA(), sparseA.getCsrColIndA(),
                getWT().front().dense.getDeviceData(),
                getWT().front().dense.getH(), &alpha, postDeviceData, postlda),
            "computing WWTA");
      };

			std::for_each(getWT().begin(), getWT().end(), func);
    }
  } catch (exception &e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
  return PV_SUCCESS;
}
}
