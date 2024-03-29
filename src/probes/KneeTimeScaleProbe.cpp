#include "KneeTimeScaleProbe.hpp"
#include "components/KneeTimeScaleController.hpp"

namespace PV {

KneeTimeScaleProbe::KneeTimeScaleProbe(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

int KneeTimeScaleProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AdaptiveTimeScaleProbe::ioParamsFillGroup(ioFlag);
   ioParam_kneeThresh(ioFlag);
   ioParam_kneeSlope(ioFlag);
   return status;
}

void KneeTimeScaleProbe::ioParam_kneeThresh(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "kneeThresh", &mKneeThresh, mKneeThresh);
}

void KneeTimeScaleProbe::ioParam_kneeSlope(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "kneeSlope", &mKneeSlope, mKneeSlope);
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
