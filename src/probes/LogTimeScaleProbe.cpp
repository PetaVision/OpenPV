#include "LogTimeScaleProbe.hpp"
#include "components/LogTimeScaleController.hpp"

namespace PV {

LogTimeScaleProbe::LogTimeScaleProbe(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

int LogTimeScaleProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AdaptiveTimeScaleProbe::ioParamsFillGroup(ioFlag);
   ioParam_logThresh(ioFlag);
   ioParam_logSlope(ioFlag);
   return status;
}

void LogTimeScaleProbe::ioParam_logThresh(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "logThresh", &mLogThresh, mLogThresh);
}

void LogTimeScaleProbe::ioParam_logSlope(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, getName(), "logSlope", &mLogSlope, mLogSlope);
}

void LogTimeScaleProbe::allocateTimeScaleController() {
   mAdaptiveTimeScaleController = new LogTimeScaleController(
         getName(),
         getNumValues(),
         mBaseMax,
         mBaseMin,
         tauFactor,
         mGrowthFactor,
         mCommunicator,
         mLogThresh,
         mLogSlope);
}
}
