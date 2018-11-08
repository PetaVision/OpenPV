#include "FailBeforeExpectedStartTimeLayer.hpp"
#include "utils/PVLog.hpp"

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer(
      char const *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer() { initialize_base(); }

int FailBeforeExpectedStartTimeLayer::initialize_base() { return PV_SUCCESS; }

void FailBeforeExpectedStartTimeLayer::initialize(
      char const *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   return PV::HyPerLayer::initialize(name, params, comm);
}

PV::Response::Status FailBeforeExpectedStartTimeLayer::updateState(double simTime, double dt) {
   FatalIf(
         simTime < mExpectedStartTime,
         "expected starting time is %f, but updateState was called with t=%f\n",
         mExpectedStartTime,
         simTime);
   return PV::HyPerLayer::updateState(simTime, dt);
}
