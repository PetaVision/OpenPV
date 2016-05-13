/*
 * PtwiseProductLayer.hpp
 *
 * The output V is the pointwise product of GSynExc and GSynInh
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#ifndef PTWISEPRODUCTLAYER_HPP_
#define PTWISEPRODUCTLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class PtwiseProductLayer : public ANNLayer {
public:
   PtwiseProductLayer(const char * name, HyPerCol * hc);
   virtual ~PtwiseProductLayer();

   virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
   // virtual int updateV();

protected:
   PtwiseProductLayer();
   int initialize(const char * name, HyPerCol * hc);

   /* static */ int doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead);

private:
   int initialize_base();
};  // end class PtwiseProductLayer

BaseObject * createPtwiseProductLayer(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* PTWISEPRODUCTLAYER_HPP_ */
