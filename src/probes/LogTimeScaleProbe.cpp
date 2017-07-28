#include "LogTimeScaleProbe.hpp"
#include "components/LogTimeScaleController.hpp"

namespace PV {

LogTimeScaleProbe::LogTimeScaleProbe(char const *name, HyPerCol *hc) { initialize(name, hc); }

int LogTimeScaleProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AdaptiveTimeScaleProbe::ioParamsFillGroup(ioFlag);
   ioParam_logThresh(ioFlag);
   ioParam_logSlope(ioFlag);
   return status;
}

void LogTimeScaleProbe::ioParam_logThresh(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "logThresh", &mLogThresh, mLogThresh);
}

void LogTimeScaleProbe::ioParam_logSlope(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "logSlope", &mLogSlope, mLogSlope);
}

void LogTimeScaleProbe::allocateTimeScaleController() {
   mAdaptiveTimeScaleController = new LogTimeScaleController(
         getName(),
         getNumValues(),
         mBaseMax,
         mBaseMin,
         tauFactor,
         mGrowthFactor,
         mWriteTimeScaleFieldnames,
         parent->getCommunicator(),
         mLogThresh,
         mLogSlope);
}
}
