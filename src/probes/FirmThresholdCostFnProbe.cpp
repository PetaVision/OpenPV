#include "FirmThresholdCostFnProbe.hpp"
#include <memory>

namespace PV {

FirmThresholdCostFnProbe::FirmThresholdCostFnProbe(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

void FirmThresholdCostFnProbe::createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   mProbeLocal = std::make_shared<FirmThresholdCostFnProbeLocal>(paramsIO);
}

void FirmThresholdCostFnProbe::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   AbstractNormProbe::initialize(paramsIO, comm);
}

} // namespace PV
