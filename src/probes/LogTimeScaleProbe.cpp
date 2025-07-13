#include "LogTimeScaleProbe.hpp"
#include "components/LogTimeScaleController.hpp"

namespace PV {

LogTimeScaleProbe::LogTimeScaleProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

int LogTimeScaleProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = AdaptiveTimeScaleProbe::ioParamsFillGroup(ioSwitch);
   ioParam_logThresh(ioSwitch);
   ioParam_logSlope(ioSwitch);
   return status;
}

void LogTimeScaleProbe::ioParam_logThresh(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "logThresh", &mLogThresh);
}

void LogTimeScaleProbe::ioParam_logSlope(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "logSlope", &mLogSlope);
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
