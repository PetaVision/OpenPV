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
   NormalizeMax(HyPerConn * callingConn);
   virtual ~NormalizeMax();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int normalizeWeights(HyPerConn * conn);

protected:
   NormalizeMax();
   int initialize(HyPerConn * callingConn);

   virtual void ioParam_minMaxTolerated(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

// Member variables
protected:
   float minMaxTolerated; // Error if abs(sum(weights)) in any patch is less than this amount.
};

} /* namespace PV */
#endif /* NORMALIZEMAX_HPP_ */
