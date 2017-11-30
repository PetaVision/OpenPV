/*
 * NormalizeSum.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZESUM_HPP_
#define NORMALIZESUM_HPP_

#include "NormalizeMultiply.hpp"

namespace PV {

class NormalizeSum : public PV::NormalizeMultiply {
   // Member functions
  public:
   NormalizeSum(const char *name, HyPerCol *hc);
   virtual ~NormalizeSum();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual int normalizeWeights() override;

  protected:
   NormalizeSum();
   int initialize(const char *name, HyPerCol *hc);

   virtual void ioParam_minSumTolerated(enum ParamsIOFlag ioFlag);

  private:
   int initialize_base();

   // Member variables
  protected:
   float mMinSumTolerated = 0.0f; // Error if any patch has abs(sum(weights)) less than this amount.

}; // class NormalizeSum

} /* namespace PV */
#endif /* NORMALIZESUM_HPP_ */
