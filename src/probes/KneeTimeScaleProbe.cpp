#include "KneeTimeScaleProbe.hpp"
#include "components/KneeTimeScaleController.hpp"

namespace PV {

KneeTimeScaleProbe::KneeTimeScaleProbe(char const *name, HyPerCol *hc) { initialize(name, hc); }

int KneeTimeScaleProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = AdaptiveTimeScaleProbe::ioParamsFillGroup(ioFlag);
   ioParam_kneeThresh(ioFlag);
   ioParam_kneeSlope(ioFlag);
   return status;
}

void KneeTimeScaleProbe::ioParam_kneeThresh(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "kneeThresh", &mKneeThresh, mKneeThresh);
}

void KneeTimeScaleProbe::ioParam_kneeSlope(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "kneeSlope", &mKneeSlope, mKneeSlope);
}

void KneeTimeScaleProbe::allocateTimeScaleController() {
   mAdaptiveTimeScaleController = new KneeTimeScaleController(
         getName(),
         getNumValues(),
         mBaseMax,
         mBaseMin,
         tauFactor,
         mGrowthFactor,
         mWriteTimeScaleFieldnames,
         parent->getCommunicator(),
         mKneeThresh,
         mKneeSlope);
}
}
