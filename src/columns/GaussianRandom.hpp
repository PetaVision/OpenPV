/*
 * GaussianRandom.hpp
 *
 *  Created on: Aug 23, 2013
 *      Author: pschultz
 */

#ifndef GAUSSIANRANDOM_HPP_
#define GAUSSIANRANDOM_HPP_

#include "Random.hpp"

struct box_muller_data {bool hasHeldValue; float heldValue;};

namespace PV {

class GaussianRandom: public PV::Random {
public:
   GaussianRandom(HyPerCol * hc, int count);
   GaussianRandom(HyPerCol * hc, const PVLayerLoc * locptr, bool isExtended);
   virtual ~GaussianRandom();

   float gaussianDist(int localIndex=0);
   float gaussianDist(int localIndex, float mean, float sigma) {return mean+gaussianDist(localIndex)*sigma;}
   void gaussianDist(float * values, int localIndex, int count=1) {for (int k=0; k<count; k++) values[k] = gaussianDist(localIndex+k);}
   void gaussianDist(float * values, int localIndex, int count, float mean, float sigma) {for (int k=0; k<count; k++) values[k] = gaussianDist(localIndex+k,mean,sigma);}

protected:
   GaussianRandom();
   int initializeFromCount(HyPerCol * hc, unsigned int count);
   int initializeFromLoc(HyPerCol* hc, const PVLayerLoc* locptr, bool isExtended);
   int initializeGaussian();

private:
   int initialize_base();

// Member variables
protected:
   struct box_muller_data * heldValues;
};

} /* namespace PV */
#endif /* GAUSSIANRANDOM_HPP_ */
