#ifndef KNEETIMESCALEPROBE_HPP_
#define KNEETIMESCALEPROBE_HPP_

#include "AdaptiveTimeScaleProbe.hpp"

namespace PV {

class KneeTimeScaleProbe : public AdaptiveTimeScaleProbe {

  public:
   virtual void ioParam_kneeThresh(ParamsIOSwitch ioSwitch);
   virtual void ioParam_kneeSlope(ParamsIOSwitch ioSwitch);

   KneeTimeScaleProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

  protected:
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void allocateTimeScaleController() override;

   double mKneeThresh = 1.0;
   double mKneeSlope  = 1.0;
};
}

#endif // KNEETIMESCALEPROBE_HPP_
