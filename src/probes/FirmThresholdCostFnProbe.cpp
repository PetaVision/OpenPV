#include "FirmThresholdCostFnProbe.hpp"
#include <memory>

namespace PV {

FirmThresholdCostFnProbe::FirmThresholdCostFnProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

void FirmThresholdCostFnProbe::createProbeLocal(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeLocal = std::make_shared<FirmThresholdCostFnProbeLocal>(params, defaults);
}

void FirmThresholdCostFnProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   AbstractNormProbe::initialize(params, defaults, comm);
}

} // namespace PV
