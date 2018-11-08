#ifndef _LOGTIMESCALEPROBE_HPP_
#define _LOGTIMESCALEPROBE_HPP_

#include "AdaptiveTimeScaleProbe.hpp"
#include <cfloat>

namespace PV {

class LogTimeScaleProbe : public AdaptiveTimeScaleProbe {

  public:
   virtual void ioParam_logThresh(enum ParamsIOFlag ioFlag);
   virtual void ioParam_logSlope(enum ParamsIOFlag ioFlag);

   LogTimeScaleProbe(char const *name, PVParams *params, Communicator *comm);

  protected:
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void allocateTimeScaleController() override;

   double mLogThresh = DBL_MAX_EXP;
   double mLogSlope  = 1.0;
};
}

#endif
