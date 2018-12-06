#include "FailBeforeExpectedStartTimeLayer.hpp"
#include "utils/PVLog.hpp"

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer(
      char const *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   initialize(name, params, comm);
}

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer() {}

void FailBeforeExpectedStartTimeLayer::initialize(
      char const *name,
      PV::PVParams *params,
      PV::Communicator *comm) {
   return PV::HyPerLayer::initialize(name, params, comm);
}

PV::Response::Status FailBeforeExpectedStartTimeLayer::checkUpdateState(double simTime, double dt) {
   FatalIf(
         simTime < mExpectedStartTime,
         "expected starting time is %f, but checkUpdateState was called with t=%f\n",
         mExpectedStartTime,
         simTime);
   return PV::HyPerLayer::checkUpdateState(simTime, dt);
}
