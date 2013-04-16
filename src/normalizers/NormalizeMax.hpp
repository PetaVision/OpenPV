/*
 * NormalizeMax.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEMAX_HPP_
#define NORMALIZEMAX_HPP_

#include "NormalizeBase.hpp"

namespace PV {

class NormalizeMax: public PV::NormalizeBase {
// Member functions
public:
   NormalizeMax(const char * name, PVParams * params);
   virtual ~NormalizeMax();

   virtual int normalizeWeights(HyPerConn * conn);

protected:
   NormalizeMax();
   int initialize(const char * name, PVParams * params);
   virtual int setParams();

   virtual void readMinSumTolerated() {minMaxTolerated = params->value(name, "minMaxTolerated", 0.0f, true/*warnIfAbsent*/);}

private:
   int initialize_base();

// Member variables
protected:
   float minMaxTolerated; // Error if abs(sum(weights)) in any patch is less than this amount.
};

} /* namespace PV */
#endif /* NORMALIZEMAX_HPP_ */
