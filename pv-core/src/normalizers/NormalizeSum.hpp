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

class NormalizeSum: public PV::NormalizeMultiply {
// Member functions
public:
   NormalizeSum(const char * name, HyPerCol * hc);
   virtual ~NormalizeSum();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int normalizeWeights();

protected:
   NormalizeSum();
   int initialize(const char * name, HyPerCol * hc);

   virtual void ioParam_minSumTolerated(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

// Member variables
protected:
   float minSumTolerated; // Error if abs(sum(weights)) in any patch is less than this amount.
}; // class NormalizeSum

BaseObject * createNormalizeSum(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* NORMALIZESUM_HPP_ */
