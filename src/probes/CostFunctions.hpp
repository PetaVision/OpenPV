#ifndef COSTFUNCTIONS_HPP_
#define COSTFUNCTIONS_HPP_

#include <cmath>

namespace PV {

class L0CostFunction {
  public:
   L0CostFunction(double threshold) : mThreshold(threshold) {}
   ~L0CostFunction() {}
   double operator()(double x) const { return std::fabs(x) > mThreshold ? 1.0 : 0.0; }

  private:
   double mThreshold;
};

class L1CostFunction {
  public:
   L1CostFunction() {}
   ~L1CostFunction() {}
   double operator()(double x) const { return std::fabs(x); }
};

class L2CostFunction {
  public:
   L2CostFunction() {}
   ~L2CostFunction() {}
   double operator()(double x) const { return x * x; }
};

class FirmThresholdCostFunction {
  public:
   FirmThresholdCostFunction(double threshold, double width) {
      mThreshPlusWidth = threshold + width;
      mMaxCost         = 0.5 * mThreshPlusWidth;
      mQuadraticCoeff  = 0.5 / mThreshPlusWidth;
   }
   ~FirmThresholdCostFunction() {}
   double operator()(double x) const {
      double absx = std::fabs(x);
      return absx < mThreshPlusWidth ? absx * (1.0 - mQuadraticCoeff * absx) : mMaxCost;
   }

  private:
   double mThreshPlusWidth;
   double mMaxCost;
   double mQuadraticCoeff;
};

} // namespace PV

#endif // COSTFUNCTIONS_HPP_
