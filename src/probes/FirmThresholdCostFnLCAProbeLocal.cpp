#include "FirmThresholdCostFnLCAProbeLocal.hpp"

namespace PV {

FirmThresholdCostFnLCAProbeLocal::FirmThresholdCostFnLCAProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

void FirmThresholdCostFnLCAProbeLocal::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   FirmThresholdCostFnProbeLocal::initialize(paramsIO);
}

void FirmThresholdCostFnLCAProbeLocal::ioParam_VThresh(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      warnUnnecessaryParameter("VThresh");
   }
}

void FirmThresholdCostFnLCAProbeLocal::ioParam_VWidth(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      warnUnnecessaryParameter("VWidth");
   }
}

void FirmThresholdCostFnLCAProbeLocal::setFirmThresholdParams(double VThresh, double VWidth) {
   mVThresh = VThresh;
   mVWidth  = VWidth;
}

void FirmThresholdCostFnLCAProbeLocal::warnUnnecessaryParameter(char const *paramName) {
   if (mParamsIO->isPresent(paramName)) {
      char const *className = mParamsIO->getKeyword().c_str();
      WarnLog().printf(
            "Parameter %s is present in the params file for %s \"%s\", but %s does not use it. "
            "Instead, %s is taken from the target layer.\n",
            paramName,
            className,
            getName_c(),
            className,
            paramName);
      // mark param as read so that presentAndNotBeenRead() doesn't trip up
      double paramValue = mParamsIO->readValue<double>(paramName, false /*warnIfAbsentFlag*/);
   }
}

} // namespace PV
