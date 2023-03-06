#include "FirmThresholdCostFnLCAProbeLocal.hpp"

namespace PV {

FirmThresholdCostFnLCAProbeLocal::FirmThresholdCostFnLCAProbeLocal(
      char const *objName,
      PVParams *params) {
   initialize(objName, params);
}

void FirmThresholdCostFnLCAProbeLocal::initialize(char const *objName, PVParams *params) {
   FirmThresholdCostFnProbeLocal::initialize(objName, params);
}

void FirmThresholdCostFnLCAProbeLocal::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      warnUnnecessaryParameter("VThresh");
   }
}

void FirmThresholdCostFnLCAProbeLocal::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      warnUnnecessaryParameter("VWidth");
   }
}

void FirmThresholdCostFnLCAProbeLocal::setFirmThresholdParams(double VThresh, double VWidth) {
   mVThresh = VThresh;
   mVWidth  = VWidth;
}

void FirmThresholdCostFnLCAProbeLocal::warnUnnecessaryParameter(char const *paramName) {
   if (getParams()->present(getName_c(), paramName)) {
      char const *className = getParams()->groupKeywordFromName(getName_c());
      WarnLog().printf(
            "Parameter %s is present in the params file for %s \"%s\", but %s does not use it. "
            "Instead, %s is taken from the target layer.\n",
            paramName,
            className,
            getName_c(),
            className,
            paramName);
      // mark param as read so that presentAndNotBeenRead() doesn't trip up
      getParams()->value(getName_c(), paramName);
   }
}

} // namespace PV
