#include "KneeTimeScaleController.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

KneeTimeScaleController::KneeTimeScaleController(
      char const *name,
      int batchWidth,
      double baseMax,
      double baseMin,
      double tauFactor,
      double growthFactor,
      Communicator const *comm,
      double kneeThresh,
      double kneeSlope)
      : AdaptiveTimeScaleController(
              name,
              batchWidth,
              baseMax,
              baseMin,
              tauFactor,
              growthFactor,
              comm) {
   mKneeThresh = kneeThresh;
   mKneeSlope  = kneeSlope;
}

std::vector<TimeScaleData> const &KneeTimeScaleController::calcTimesteps(std::vector<double> const &timeScales) {
   AdaptiveTimeScaleController::calcTimesteps(timeScales);
   // Updates mTimeScaleInfo, mOldTimeScaleInfo
   // Now scale timescalemax if it's above the knee
   for (int i = 0; i < mBatchWidth; ++i) {
      if (mTimeScaleInfo[i].mTimeScaleMax > mKneeThresh) {
         mTimeScaleInfo[i].mTimeScaleMax =
               mOldTimeScaleInfo[i].mTimeScaleMax
               + (mTimeScaleInfo[i].mTimeScaleMax - mOldTimeScaleInfo[i].mTimeScaleMax)
                       * mKneeSlope;
         // Cap our timescale to the newly calculated max
         mTimeScaleInfo[i].mTimeScale =
               std::min(mTimeScaleInfo[i].mTimeScale, mTimeScaleInfo[i].mTimeScaleMax);
      }
   }
   return mTimeScaleInfo;
}

} // namespace PV
