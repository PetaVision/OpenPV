#include "FailBeforeExpectedStartTimeLayer.hpp"
#include "utils/PVLog.hpp"

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer(
      char const *name,
      PV::HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer() { initialize_base(); }

int FailBeforeExpectedStartTimeLayer::initialize_base() { return PV_SUCCESS; }

int FailBeforeExpectedStartTimeLayer::initialize(char const *name, PV::HyPerCol *hc) {
   return PV::HyPerLayer::initialize(name, hc);
}

#ifdef PV_USE_CUDA
PV::Response::Status FailBeforeExpectedStartTimeLayer::updateStateGpu(double simTime, double dt) {
   return updateState(simTime, dt);
}
#endif // PV_USE_CUDA

PV::Response::Status FailBeforeExpectedStartTimeLayer::updateState(double simTime, double dt) {
   FatalIf(
         simTime < mExpectedStartTime,
         "expected starting time is %f, but updateState was called with t=%f\n",
         mExpectedStartTime,
         simTime);
   return PV::HyPerLayer::updateState(simTime, dt);
}
