/*
 * NormalizeSum.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZESUM_HPP_
#define NORMALIZESUM_HPP_

#include "NormalizeBase.hpp"

namespace PV {

class NormalizeSum: public PV::NormalizeBase {
// Member functions
public:
   NormalizeSum(const char * name, PVParams * params);
   virtual ~NormalizeSum();

   virtual int normalizeWeights(HyPerConn * conn);

protected:
   NormalizeSum();
   int initialize(const char * name, PVParams * params);
   virtual int setParams();

   virtual void readMinSumTolerated() {minSumTolerated = params->value(name, "minSumTolerated", 0.0f, true/*warnIfAbsent*/);}

private:
   int initialize_base();

// Member variables
protected:
   float minSumTolerated; // Error if abs(sum(weights)) in any patch is less than this amount.
};

} /* namespace PV */
#endif /* NORMALIZESUM_HPP_ */
