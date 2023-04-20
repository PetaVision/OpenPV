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
      Communicator const *comm,
      double logThresh,
      double logSlope)
      : AdaptiveTimeScaleController(
              name,
              batchWidth,
              baseMax,
              baseMin,
              tauFactor,
              growthFactor,
              comm) {
   mLogThresh = logThresh;
   mLogSlope  = -(logThresh - mBaseMax) / log(logSlope);
}

std::vector<TimeScaleData> const &LogTimeScaleController::calcTimesteps(std::vector<double> const &timeScales) {
   FatalIf(
         static_cast<int>(timeScales.size()) != mBatchWidth,
         "new timeScaleData has different size than old timeScaleData (%d versus %d)\n",
         static_cast<int>(mTimeScaleInfo.size()),
         mBatchWidth);
   mOldTimeScaleInfo = mTimeScaleInfo;
   for (int b = 0; b < mBatchWidth; ++b) {
      mTimeScaleInfo[b].mTimeScaleTrue = timeScales[b];
      double E_dt = mTimeScaleInfo[b].mTimeScaleTrue;
      double E_0  = mOldTimeScaleInfo[b].mTimeScaleTrue;
      if (E_dt == E_0) { continue; }

      double dE_dt_scaled = (E_0 - E_dt) / mTimeScaleInfo[b].mTimeScale;

      if ((dE_dt_scaled <= 0.0) or (E_0 <= 0.0) or (E_dt <= 0.0)) {
         mTimeScaleInfo[b].mTimeScale    = mBaseMin;
         mTimeScaleInfo[b].mTimeScaleMax = mBaseMax;
      }
      else {
         double tau_eff_scaled = E_0 / dE_dt_scaled;
         mTimeScaleInfo[b].mTimeScale = mTauFactor * tau_eff_scaled;
         if (mTimeScaleInfo[b].mTimeScale >= mTimeScaleInfo[b].mTimeScaleMax) {
            double growthFactor =
                  mGrowthFactor * exp(-(mTimeScaleInfo[b].mTimeScaleMax - mBaseMax ) / mLogSlope);
            mTimeScaleInfo[b].mTimeScale = mTimeScaleInfo[b].mTimeScaleMax;
            mTimeScaleInfo[b].mTimeScaleMax *= (1 + growthFactor);
         }
      }
   }
   return mTimeScaleInfo;
}

} // namespace PV
