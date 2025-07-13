#include "FailBeforeExpectedStartTimeLayer.hpp"
#include "utils/PVLog.hpp"

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer(
      std::shared_ptr<PV::ParamGroup> params,
      std::shared_ptr<PV::ParamGroup> defaults,
      PV::Communicator const *comm) {
   initialize(params, defaults, comm);
}

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer() {}

void FailBeforeExpectedStartTimeLayer::initialize(
      std::shared_ptr<PV::ParamGroup> params,
      std::shared_ptr<PV::ParamGroup> defaults,
      PV::Communicator const *comm) {
   return PV::HyPerLayer::initialize(params, defaults, comm);
}

PV::Response::Status FailBeforeExpectedStartTimeLayer::checkUpdateState(double simTime, double dt) {
   FatalIf(
         simTime < mExpectedStartTime,
         "expected starting time is %f, but checkUpdateState was called with t=%f\n",
         mExpectedStartTime,
         simTime);
   return PV::HyPerLayer::checkUpdateState(simTime, dt);
}
