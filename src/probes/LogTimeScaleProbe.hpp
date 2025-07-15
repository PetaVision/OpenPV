#ifndef LOGTIMESCALEPROBE_HPP_
#define LOGTIMESCALEPROBE_HPP_

#include "AdaptiveTimeScaleProbe.hpp"
#include <cfloat>

namespace PV {

class LogTimeScaleProbe : public AdaptiveTimeScaleProbe {

  public:
   virtual void ioParam_logThresh(ParamsIOSwitch ioSwitch);
   virtual void ioParam_logSlope(ParamsIOSwitch ioSwitch);

   LogTimeScaleProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  protected:
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;
   virtual void allocateTimeScaleController() override;

   double mLogThresh = DBL_MAX_EXP;
   double mLogSlope  = 1.0;
};
}

#endif // LOGTIMESCALEPROBE_HPP_
