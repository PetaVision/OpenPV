#ifndef KNEETIMESCALECONTROLLER_HPP_
#define KNEETIMESCALECONTROLLER_HPP_

#include "AdaptiveTimeScaleController.hpp"

namespace PV {

class KneeTimeScaleController : public AdaptiveTimeScaleController {
  public:
   KneeTimeScaleController(
         char const *name,
         int batchWidth,
         double baseMax,
         double baseMin,
         double tauFactor,
         double growthFactor,
         Communicator const *comm,
         double kneeThresh,
         double kneeSlope);

   virtual std::vector<TimeScaleData> const &calcTimesteps(std::vector<double> const &timeScales) override;

  protected:
   double mKneeThresh = 1.0;
   double mKneeSlope  = 1.0;
};
}

#endif // KNEETIMESCALECONTROLLER_HPP_
