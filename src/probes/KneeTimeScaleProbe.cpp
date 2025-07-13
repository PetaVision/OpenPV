#include "KneeTimeScaleProbe.hpp"
#include "components/KneeTimeScaleController.hpp"

namespace PV {

KneeTimeScaleProbe::KneeTimeScaleProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

int KneeTimeScaleProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = AdaptiveTimeScaleProbe::ioParamsFillGroup(ioSwitch);
   ioParam_kneeThresh(ioSwitch);
   ioParam_kneeSlope(ioSwitch);
   return status;
}

void KneeTimeScaleProbe::ioParam_kneeThresh(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "kneeThresh", &mKneeThresh);
}

void KneeTimeScaleProbe::ioParam_kneeSlope(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "kneeSlope", &mKneeSlope);
}

void KneeTimeScaleProbe::allocateTimeScaleController() {
   mAdaptiveTimeScaleController = new KneeTimeScaleController(
         getName(),
         getNumValues(),
         mBaseMax,
         mBaseMin,
         tauFactor,
         mGrowthFactor,
         mCommunicator,
         mKneeThresh,
         mKneeSlope);
}

} // namespace PV
