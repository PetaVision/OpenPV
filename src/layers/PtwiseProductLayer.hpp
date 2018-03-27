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
   PtwiseProductLayer(const char *name, HyPerCol *hc);
   virtual ~PtwiseProductLayer();

   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status updateState(double timef, double dt) override;

  protected:
   PtwiseProductLayer();
   int initialize(const char *name, HyPerCol *hc);

   int updateState(
         double timef,
         double dt,
         const PVLayerLoc *loc,
         float *A,
         float *V,
         int num_channels,
         float *gSynHead);

  private:
   int initialize_base();
}; // end class PtwiseProductLayer

} // end namespace PV

#endif /* PTWISEPRODUCTLAYER_HPP_ */
