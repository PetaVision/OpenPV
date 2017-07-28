#include "LogTimeScaleController.hpp"
#include <cmath>

namespace PV {

LogTimeScaleController::LogTimeScaleController(
      char const *name,
      int batchWidth,
      double baseMax,
      double baseMin,
      double tauFactor,
      double growthFactor,
      bool writeTimeScaleFieldnames,
      Communicator *comm,
      double logThresh,
      double logSlope)
      : AdaptiveTimeScaleController(
              name,
              batchWidth,
              baseMax,
              baseMin,
              tauFactor,
              growthFactor,
              writeTimeScaleFieldnames,
              comm) {
   mLogThresh = logThresh;
   mLogSlope  = -(logThresh - mBaseMax) / log(logSlope);
}

std::vector<double>
LogTimeScaleController::calcTimesteps(double timeValue, std::vector<double> const &rawTimeScales) {
   mOldTimeScaleInfo             = mTimeScaleInfo;
   mTimeScaleInfo.mTimeScaleTrue = rawTimeScales;
   for (int b = 0; b < mBatchWidth; b++) {
      double E_dt         = mTimeScaleInfo.mTimeScaleTrue[b];
      double E_0          = mOldTimeScaleInfo.mTimeScaleTrue[b];
      double dE_dt_scaled = (E_0 - E_dt) / mTimeScaleInfo.mTimeScale[b];

      if (E_dt == E_0) {
         continue;
      }

      if ((dE_dt_scaled <= 0.0) || (E_0 <= 0) || (E_dt <= 0)) {
         mTimeScaleInfo.mTimeScale[b]    = mBaseMin;
         mTimeScaleInfo.mTimeScaleMax[b] = mBaseMax;
      }
      else {
         double tau_eff_scaled = E_0 / dE_dt_scaled;

         // dt := mTimeScaleMaxBase * tau_eff
         mTimeScaleInfo.mTimeScale[b] = mTauFactor * tau_eff_scaled;
         if (mTimeScaleInfo.mTimeScale[b] >= mTimeScaleInfo.mTimeScaleMax[b]) {
            double growthFactor =
                  mGrowthFactor * exp(-(mTimeScaleInfo.mTimeScaleMax[b] - mBaseMax) / mLogSlope);
            mTimeScaleInfo.mTimeScale[b]    = mTimeScaleInfo.mTimeScaleMax[b];
            mTimeScaleInfo.mTimeScaleMax[b] = (1 + growthFactor) * mTimeScaleInfo.mTimeScaleMax[b];
         }
      }
   }
   return mTimeScaleInfo.mTimeScale;
}
}
