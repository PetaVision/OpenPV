#ifndef LOGTIMESCALECONTROLLER_HPP_
#define LOGTIMESCALECONTROLLER_HPP_

#include "AdaptiveTimeScaleController.hpp"
#include <cfloat>

namespace PV {

class LogTimeScaleController : public AdaptiveTimeScaleController {
  public:
   LogTimeScaleController(
         char const *name,
         int batchWidth,
         double baseMax,
         double baseMin,
         double tauFactor,
         double growthFactor,
         Communicator const *comm,
         double logThresh,
         double logSlope);

   virtual std::vector<TimeScaleData> const &calcTimesteps(std::vector<double> const &timeScales) override;

  protected:
   double mLogThresh = DBL_MAX_EXP;
   double mLogSlope  = 1.0;
};
}

#endif // LOGTIMESCALECONTROLLER_HPP_
