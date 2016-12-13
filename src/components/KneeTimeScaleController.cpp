#include "KneeTimeScaleController.hpp"

namespace PV {

KneeTimeScaleController::KneeTimeScaleController(
      char const *name,
      int batchWidth,
      double baseMax,
      double baseMin,
      double tauFactor,
      double growthFactor,
      bool writeTimeScales,
      bool writeTimeScaleFieldnames,
      Communicator *comm,
      bool verifyWrites,
      double kneeThresh,
      double kneeSlope) : AdaptiveTimeScaleController(
         name,
         batchWidth,
         baseMax,
         baseMin,
         tauFactor,
         growthFactor,
         writeTimeScales,
         writeTimeScaleFieldnames,
         comm,
         verifyWrites) {
   mKneeThresh = kneeThresh;
   mKneeSlope = kneeSlope;
}

std::vector<double> 
KneeTimeScaleController::calcTimesteps(
      double timeValue, 
      std::vector<double> const &rawTimeScales) {
   std::vector<double> timeScales(AdaptiveTimeScaleController::calcTimesteps(timeValue, rawTimeScales));

   for (auto& ts : timeScales) {
      ts = (ts < mKneeThresh)
         ? ts
         : mKneeThresh + (ts - mKneeThresh) * mKneeSlope;
   }
   return timeScales;
}

}
