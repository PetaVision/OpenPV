#include "FailBeforeExpectedStartTimeLayer.hpp"
#include "utils/PVLog.hpp"

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm) {
   initialize(paramsIO, comm);
}

FailBeforeExpectedStartTimeLayer::FailBeforeExpectedStartTimeLayer() {}

void FailBeforeExpectedStartTimeLayer::initialize(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm) {
   return PV::HyPerLayer::initialize(paramsIO, comm);
}

PV::Response::Status FailBeforeExpectedStartTimeLayer::checkUpdateState(double simTime, double dt) {
   FatalIf(
         simTime < mExpectedStartTime,
         "expected starting time is %f, but checkUpdateState was called with t=%f\n",
         mExpectedStartTime,
         simTime);
   return PV::HyPerLayer::checkUpdateState(simTime, dt);
}
