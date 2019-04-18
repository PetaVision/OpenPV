#include "KneeTimeScaleController.hpp"

namespace PV {

KneeTimeScaleController::KneeTimeScaleController(
      char const *name,
      int batchWidth,
      double baseMax,
      double baseMin,
      double tauFactor,
      double growthFactor,
      bool writeTimeScaleFieldnames,
      Communicator *comm,
      double kneeThresh,
      double kneeSlope)
      : AdaptiveTimeScaleController(
              name,
              batchWidth,
              baseMax,
              baseMin,
              tauFactor,
              growthFactor,
              writeTimeScaleFieldnames,
              comm) {
   mKneeThresh = kneeThresh;
   mKneeSlope  = kneeSlope;
}

std::vector<double>
KneeTimeScaleController::calcTimesteps(double timeValue, std::vector<double> const &rawTimeScales) {

   std::vector<double> timeScales(
         AdaptiveTimeScaleController::calcTimesteps(timeValue, rawTimeScales));

   for (int i = 0; i < timeScales.size(); ++i) {
      // Scale timescalemax if it's above the knee
      if (mTimeScaleInfo.mTimeScaleMax[i] > mKneeThresh) {
         mTimeScaleInfo.mTimeScaleMax[i] =
               mOldTimeScaleInfo.mTimeScaleMax[i]
               + (mTimeScaleInfo.mTimeScaleMax[i] - mOldTimeScaleInfo.mTimeScaleMax[i])
                       * mKneeSlope;
         // Cap our timescale to the newly calculated max
         timeScales[i] = std::min(timeScales[i], mTimeScaleInfo.mTimeScaleMax[i]);
      }
   }
   return timeScales;
}
}
