#include <functional>
#include "ANNLayerGPU.hpp"

namespace GPULCA {

ANNLayerGPU::ANNLayerGPU() {
  initialize_base();
  initialize();
}

ANNLayerGPU::ANNLayerGPU(const char* name, PV::HyPerCol* hc) {
  ANNLayer::initialize(name, hc);
  initialize_base();
  initialize();
}

ANNLayerGPU::~ANNLayerGPU() {
  /*  cuSparse destroy */
  cusparseStatusDestructorCheck(cusparseDestroy(cusparseHandle),
                                "destory handle");
  cusparseStatusDestructorCheck(cusparseDestroyMatDescr(cusparseMatDescr),
                                "destory matrix descriptor");
}

void ANNLayerGPU::initialize_base() {
  /*  force it to be false here, because there is no sparse convolution function
   * yet.  */
  sparseLayer = false;  // when it is false, the GSyn and V use dense format,
                        // activiey uses both dense and sparse format; when it
                        // is false, everything uses sparse format.
}

int ANNLayerGPU::initialize() {
  /*  CUDA initialization  */
  /*  cuSparse initialization */
  try {
    cusparseStatusCheck(cusparseCreate(&cusparseHandle), "create handle");
    cusparseStatusCheck(cusparseCreateMatDescr(&cusparseMatDescr),
                        "create matrix descriptor");
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

int ANNLayerGPU::publish(PV::Communicator* comm, double time) {
  publish_timer->start();

  bool mirroring = useMirrorBCs();
  mirroring = mirroring ? (getLastUpdateTime() >= getParent()->simulationTime())
                        : false;

  activity.dense.device2Host();
  const PVLayerLoc* l = getLayerLoc();
  PVLayerCube a = {.size = getCLayer()->activity->size,
                   .numItems = getCLayer()->activity->numItems,
                   .data = activity.dense.getHostData(),
                   .padding = {},
                   .loc = {l->nbatch,
                           l->nx,
                           l->ny,
                           l->nf,
                           l->nbatchGlobal,
                           l->nxGlobal,
                           l->nyGlobal,
                           l->kb0,
                           l->kx0,
                           l->ky0,
                           {l->halo.lt, l->halo.rt, l->halo.dn, l->halo.up}},
                   .isSparse = getCLayer()->activity->isSparse,
                   .numActive = getCLayer()->activity->numActive,
                   .activeIndices = getCLayer()->activity->activeIndices};

  if (mirroring) {
    mirrorInteriorToBorder(&a, &a);
  }

  int status = publisher->publish(time, lastUpdateTime, &a);
  publish_timer->stop();
  return status;
}

int ANNLayerGPU::allocateGSyn() {
  try {
    if (getNumChannels() > 0) {
      const PVLayerLoc* loc = getLayerLoc();
      MatrixInfo matrixParams = {.n = 1,
                                 .height = loc->ny,
                                 .width = loc->nx,
                                 .channel = loc->nf,
                                 .layout = NCHW},
                 transposedmatrixParams = {.n = 1,
                                           .height = loc->ny,
                                           .width = loc->nx,
                                           .channel = loc->nf,
                                           .layout = NHWC};

      std::function<void(PVCudaWrapper<pvdata_t>&)> initFunc;

      if (getSparseFlag())
        initFunc = [&](PVCudaWrapper<pvdata_t>& x) {
          x.sparse.init(transposedmatrixParams, &cusparseHandle,
                        &cusparseMatDescr, true);
        };
      else
        initFunc = [&](PVCudaWrapper<pvdata_t>& x) {
          x.dense.init(matrixParams);
          x.sparse.init(transposedmatrixParams, &cusparseHandle,
                        &cusparseMatDescr, true);
        };
      GSyn.resize(getNumChannels());
      std::for_each(GSyn.begin(), GSyn.end(), initFunc);
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

int ANNLayerGPU::allocateV() {
  HyPerLayer::allocateV();
  const PVLayerLoc* loc = getLayerLoc();
  MatrixInfo matrixParams = {.n = 1,
                             .height = loc->ny,
                             .width = loc->nx,
                             .channel = loc->nf,
                             .layout = NCHW},
             transposedmatrixParams = {.n = 1,
                                       .height = loc->ny,
                                       .width = loc->nx,
                                       .channel = loc->nf,
                                       .layout = NHWC};

  try {
    if (getSparseFlag())
      V.sparse.init(transposedmatrixParams, &cusparseHandle, &cusparseMatDescr,
                    true);
    else
      V.dense.init(matrixParams);
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

int ANNLayerGPU::allocateActivity() {
  const PVLayerLoc* loc = getLayerLoc();
  MatrixInfo matrixParams = {.n = 1,
                             .height = loc->ny,
                             .width = loc->nx,
                             .channel = loc->nf,
                             .layout = NCHW},
             transposedmatrixParams = {.n = 1,
                                       .height = loc->ny,
                                       .width = loc->nx,
                                       .channel = loc->nf,
                                       .layout = NHWC};
  try {
    if (getSparseFlag()) {
      activity.dense.init(matrixParams);
      activity.sparse.init(transposedmatrixParams, &cusparseHandle,
                           &cusparseMatDescr, true);
    } else {
      activity.dense.init(matrixParams);
    }

  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

int ANNLayerGPU::resetGSynBuffers(double timef, double dt) {
  try {
    if (!getSparseFlag()) {
      auto resetFunc = [](PVCudaWrapper<pvdata_t>& x) {
        x.dense.setDenseMatrixDeviceData(0);
      };
      std::for_each(GSyn.begin(), GSyn.end(), resetFunc);
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

int ANNLayerGPU::setActivity() {
  pvdata_t VTW = getVThresh() + getVWidth();
  pvdata_t tanTheta = tan(atan((VTW - getAShift()) / VTW));

  try {
    if (getSparseFlag()) {
      cerr << "No sparse to sparse activation funciton yet." << endl;
      return PV_FAILURE;
    } else {
      activationFunc(V.dense.getDeviceData(), activity.dense.getDeviceData(),
                     V.dense.getSize(), getVThresh(), getAMin(), getAMax(),
                     getAShift(), VTW, tanTheta);
    }

    cudaStatusCheck("computing A");

    getActivity().dense2sparse();

    if (getActivity().sparse.getNNZ() == 0)
      throw std::runtime_error("Activities are all zeros.");
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

int ANNLayerGPU::initializeV() {
  ANNLayer::initializeV();

  try {
    if (getSparseFlag()) {
      cerr << "No sparse V yet.\n";
      return PV_FAILURE;
    } else {
      V.dense.getCudaVector().setDeviceData(V.dense.getCudaVector().getSize(),
                                            HyPerLayer::getV());
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }

  return PV_SUCCESS;
}

int ANNLayerGPU::updateState(double timef, double dt) {
  /* Compute GSyn sum */
  try {
    if (getSparseFlag()) {
      cerr << "No sparse GSyn implementation yet.\n";
      return PV_FAILURE;
    } else {
      V.dense.getCudaVector().setDeviceData(
          getGSyn().front().dense.getCudaVector());
      std::vector<PVCudaWrapper<pvdata_t>>::iterator it = getGSyn().begin();
      std::advance(it, 1);
      auto sumFunction = [&](PVCudaWrapper<pvdata_t>& x) {
        pvdata_t alpha = 1, beta = 1;
        CudaMatrixAdd(V.dense.getSize(), alpha,
                                x.dense.getDeviceData(), beta,
                                V.dense.getDeviceData());
        cudaStatusCheck("computing GSyn summation");
      };

      std::for_each(it, getGSyn().end(), sumFunction);

      setActivity();
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    return PV_FAILURE;
  }
  return PV_SUCCESS;
}

int ANNLayerGPU::writeActivity(double timed) {
	PV::DataStore* store = publisher->dataStore();

  // copy activity to datastore.
	activity.dense.device2Host();
	memcpy(store->buffer(0), activity.dense.getHostData(), activity.dense.getSize() * sizeof(pvdata_t));

  int status = PV::writeActivity(outputStateStream, parent->getCommunicator(),
                                 timed, store, getLayerLoc());

  if (status == PV_SUCCESS) {
    status = incrementNBands(&writeActivityCalls);
  }
  return status;
}
}
