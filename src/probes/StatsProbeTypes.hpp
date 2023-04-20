#ifndef STATSBUFFERTYPE_HPP_
#define STATSBUFFERTYPE_HPP_

#include <cfloat>
#include <cmath>

namespace PV {

enum class StatsBufferType { V, A };

struct LayerStats {
   LayerStats() { reset(); }
   void reset() {
      mSum        = 0.0;
      mSumSquared = 0.0;
      mMin        = FLT_MAX;
      mMax        = -FLT_MAX;
      mNumNeurons = 0;
      mNumNonzero = 0; // number outside of threshold
   }
   double mSum;
   double mSumSquared;
   float mMin;
   float mMax;
   int mNumNeurons;
   int mNumNonzero; // number outside of threshold

   double average() const { return mSum / static_cast<double>(mNumNeurons); }

   double sigma() const {
      double dNumNeurons = static_cast<double>(mNumNeurons);
      double average     = mSum / dNumNeurons;
      return std::sqrt(mSumSquared / dNumNeurons - average * average);
   }

   void derivedStats(double &average, double &sigma) const {
      double dNumNeurons = static_cast<double>(mNumNeurons);
      average            = mSum / dNumNeurons;
      sigma              = std::sqrt(mSumSquared / dNumNeurons - average * average);
   }
};

} // namespace PV

#endif // STATSBUFFERTYPE_HPP_
