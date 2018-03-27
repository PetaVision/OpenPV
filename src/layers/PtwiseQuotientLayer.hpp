/*
 * PtwiseQuotientLayer.hpp
 *
 * The output V is the pointwise division of GSynExc by GSynInh
 * no checking for zero divisors is performed
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 * created by gkenyon, 06/2016g
 * based on PtwiseProductLayer Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#ifndef PTWISEQUOTIENTLAYER_HPP_
#define PTWISEQUOTIENTLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

class PtwiseQuotientLayer : public ANNLayer {
  public:
   PtwiseQuotientLayer(const char *name, HyPerCol *hc);
   virtual ~PtwiseQuotientLayer();

   virtual Response::Status allocateDataStructures() override;
   virtual Response::Status updateState(double timef, double dt) override;

  protected:
   PtwiseQuotientLayer();
   int initialize(const char *name, HyPerCol *hc);

   /* static */ void doUpdateState(
         double timef,
         double dt,
         const PVLayerLoc *loc,
         float *A,
         float *V,
         int num_channels,
         float *gSynHead);

  private:
   int initialize_base();
}; // end class PtwiseQuotientLayer

} // end namespace PV

#endif /* PTWISEQUOTIENTLAYER_HPP_ */
