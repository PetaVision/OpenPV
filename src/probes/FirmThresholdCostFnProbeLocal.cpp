#include "FirmThresholdCostFnProbeLocal.hpp"
#include <memory>

namespace PV {

FirmThresholdCostFnProbeLocal::FirmThresholdCostFnProbeLocal(
      char const *objName,
      PVParams *params) {
   initialize(objName, params);
}

std::shared_ptr<FirmThresholdCostFunctionSum const>
FirmThresholdCostFnProbeLocal::createCostFunctionSum() {
   auto costFunction = std::make_shared<FirmThresholdCostFunction>(mVThresh, mVWidth);
   auto costFnSum    = std::make_shared<FirmThresholdCostFunctionSum>(costFunction);
   return costFnSum;
}

void FirmThresholdCostFnProbeLocal::initialize(char const *objName, PVParams *params) {
   BaseFirmThresholdCostFnProbeLocal::initialize(objName, params);
}

void FirmThresholdCostFnProbeLocal::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   getParams()->ioParamValue(ioFlag, getName_c(), "VThresh", &mVThresh, mVThresh);
}

void FirmThresholdCostFnProbeLocal::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   getParams()->ioParamValue(ioFlag, getName_c(), "VWidth", &mVWidth, mVWidth);
}

void FirmThresholdCostFnProbeLocal::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   BaseFirmThresholdCostFnProbeLocal::ioParamsFillGroup(ioFlag);
   ioParam_VThresh(ioFlag);
   ioParam_VWidth(ioFlag);
}

} // namespace PV
