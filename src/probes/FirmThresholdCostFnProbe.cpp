#include "FirmThresholdCostFnProbe.hpp"
#include <memory>

namespace PV {

FirmThresholdCostFnProbe::FirmThresholdCostFnProbe(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

void FirmThresholdCostFnProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<FirmThresholdCostFnProbeLocal>(name, params);
}

void FirmThresholdCostFnProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   AbstractNormProbe::initialize(name, params, comm);
}

} // namespace PV
