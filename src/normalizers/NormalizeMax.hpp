/*
 * NormalizeMax.hpp
 *
 *  Created on: Apr 8, 2013
 *      Author: pschultz
 */

#ifndef NORMALIZEMAX_HPP_
#define NORMALIZEMAX_HPP_

#include "NormalizeMultiply.hpp"

namespace PV {

class NormalizeMax: public PV::NormalizeMultiply {
// Member functions
public:
   NormalizeMax(const char * name, HyPerCol * hc);
   virtual ~NormalizeMax();

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int normalizeWeights();

protected:
   NormalizeMax();
   int initialize(const char * name, HyPerCol * hc);

   virtual void ioParam_minMaxTolerated(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();

// Member variables
protected:
   float minMaxTolerated; // Error if abs(sum(weights)) in any patch is less than this amount.
};

BaseObject * createNormalizeMax(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* NORMALIZEMAX_HPP_ */
