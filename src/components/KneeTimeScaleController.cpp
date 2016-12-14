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

   for (int i = 0; i < timeScales.size(); ++i) {
      // Scale timescalemax if it's above the knee
      double tsMax = mTimeScaleInfo.mTimeScaleMax.at(i);
   
      tsMax = (tsMax < mKneeThresh)
         ? tsMax
         : mKneeThresh + (tsMax - mKneeThresh) * mKneeSlope;
      mTimeScaleInfo.mTimeScaleMax.at(i) = tsMax;

      // Cap our timescale to the newly calculated max
      timeScales.at(i) = std::min(timeScales.at(i), tsMax);
   }
   return timeScales;
}

}
