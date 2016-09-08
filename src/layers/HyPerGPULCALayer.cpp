#include <numeric>
#include <cmath>
#include "HyPerGPULCALayer.hpp"

namespace GPULCA {

HyPerGPULCALayer::HyPerGPULCALayer() {
  initialize_base();
  init();
}

HyPerGPULCALayer::HyPerGPULCALayer(const char* name, PV::HyPerCol* hc)
    : ANNLayerGPU(name, hc) {
  initialize_base();
  init();
}

HyPerGPULCALayer::~HyPerGPULCALayer() {
  /*  cuBlas destroy */
  cublasStatusDestructorCheck(cublasDestroy(cublasHandle), "destroy handle");
}

int HyPerGPULCALayer::initialize_base() {
  timeConstantTau = 1.0;
  return PV_SUCCESS;
}

void HyPerGPULCALayer::ioParam_timeConstantTau(enum PV::ParamsIOFlag ioFlag) {
  parent->ioParamValue(ioFlag, name, "timeConstantTau", &timeConstantTau,
                       timeConstantTau, true /*warnIfAbsent*/);
}

int HyPerGPULCALayer::ioParamsFillGroup(enum PV::ParamsIOFlag ioFlag) {
  int status = ANNLayer::ioParamsFillGroup(ioFlag);
  ioParam_timeConstantTau(ioFlag);
  return status;
}

int HyPerGPULCALayer::init() {
  const PVLayerLoc* loc = getLayerLoc();
  MatrixInfo matrixParams = {.n = 1,
                             .height = loc->ny,
                             .width = loc->nx,
                             .channel = loc->nf,
                             .layout = NCHW},
             transposedMatrixParams = {.n = 1,
                                       .height = loc->ny,
                                       .width = loc->nx,
                                       .channel = loc->nf,
                                       .layout = NHWC};

  try {
    if (getSparseFlag())
      UDot.sparse.init(transposedMatrixParams, &getCusparseHandle(),
                       &getCusparseMatDescr(), true);
    else
      UDot.dense.init(matrixParams);
    /*  cuBlas initialization */
    cublasStatusCheck(cublasCreate(&cublasHandle), "create handle");
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

/*  reimplement the virtual functions */
int HyPerGPULCALayer::updateState(double time, double dt) {
  /* Compute GSyn sum */
  try {
    if (getSparseFlag()) {
      cerr << "No sparse GSyn implementation yet.\n";
      return PV_FAILURE;
    } else {
      UDot.dense.getCudaVector().setDeviceData(
          getGSyn().front().dense.getCudaVector());
      std::vector<PVCudaWrapper<pvdata_t>>::iterator it = getGSyn().begin();
      std::advance(it, 1);
      auto sumFunction = [&](PVCudaWrapper<pvdata_t>& x) {
        pvdata_t alpha = 1, beta = 1;
        CudaMatrixAdd<pvdata_t>(UDot.dense.getSize(), alpha,
                                x.dense.getDeviceData(), beta,
                                UDot.dense.getDeviceData());
        cudaStatusCheck("computing GSyn summation");
      };

      std::for_each(it, getGSyn().end(), sumFunction);

      /*  GSyn + sparse A */
      getActivity().sparse.sparseMatrix2Vector();
      pvdata_t a = 1;
      cusparseStatus_t cusparseStatus =
          cusparseSaxpyi(getCusparseHandle(), getActivity().sparse.getNNZ(), &a,
                         getActivity().sparse.getCooCsrValA(),
                         getActivity().sparse.getVecInd(),
                         UDot.dense.getDeviceData(), CUSPARSE_INDEX_BASE_ZERO);
      cusparseStatusCheck(cusparseStatus, "computing GSyn + sparse A");

      /*  update V */
      pvdata_t alpha = 1 / getChannelTimeConst(CHANNEL_EXC);
      pvdata_t beta = 1 - alpha;
      CudaMatrixAdd(getV().dense.getSize(), alpha, UDot.dense.getDeviceData(),
                    beta, getV().dense.getDeviceData());
      cudaStatusCheck("updating V");

      ANNLayerGPU::setActivity();
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
  return PV_SUCCESS;
}
}
