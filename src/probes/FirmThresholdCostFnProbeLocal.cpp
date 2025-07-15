#include "FirmThresholdCostFnProbeLocal.hpp"
#include <memory>

namespace PV {

FirmThresholdCostFnProbeLocal::FirmThresholdCostFnProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

std::shared_ptr<FirmThresholdCostFunctionSum const>
FirmThresholdCostFnProbeLocal::createCostFunctionSum() {
   auto costFunction = std::make_shared<FirmThresholdCostFunction>(mVThresh, mVWidth);
   auto costFnSum    = std::make_shared<FirmThresholdCostFunctionSum>(costFunction);
   return costFnSum;
}

void FirmThresholdCostFnProbeLocal::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   BaseFirmThresholdCostFnProbeLocal::initialize(paramsIO);
}

void FirmThresholdCostFnProbeLocal::ioParam_VThresh(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "VThresh", &mVThresh);
}

void FirmThresholdCostFnProbeLocal::ioParam_VWidth(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "VWidth", &mVWidth);
}

void FirmThresholdCostFnProbeLocal::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   BaseFirmThresholdCostFnProbeLocal::ioParamsFillGroup(ioSwitch);
   ioParam_VThresh(ioSwitch);
   ioParam_VWidth(ioSwitch);
}

} // namespace PV
