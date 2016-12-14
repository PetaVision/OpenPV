#ifndef _KNEETIMESCALEPROBE_HPP_
#define _KNEETIMESCALEPROBE_HPP_

#include "AdaptiveTimeScaleProbe.hpp"

namespace PV {

class KneeTimeScaleProbe : public AdaptiveTimeScaleProbe {

  public:
   virtual void ioParam_kneeThresh(enum ParamsIOFlag ioFlag);
   virtual void ioParam_kneeSlope(enum ParamsIOFlag ioFlag);

   KneeTimeScaleProbe(char const *name, HyPerCol *hc);

  protected:
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void allocateTimeScaleController();

   double mKneeThresh = 1.0;
   double mKneeSlope  = 1.0;
};

}

#endif
