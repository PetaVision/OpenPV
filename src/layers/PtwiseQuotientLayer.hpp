/*
 * PtwiseQuotientLayer.hpp
 *
 * created by gkenyon, 06/2016g
 * based on PtwiseProductLayer Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#ifndef PTWISEQUOTIENTLAYER_HPP_
#define PTWISEQUOTIENTLAYER_HPP_

#include "ANNLayer.hpp"

namespace PV {

/**
 * The output V is the pointwise division of GSynExc by GSynInh
 * no checking for zero divisors is performed
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 * The activity buffer is an ANNActivityBuffer.
 */
class PtwiseQuotientLayer : public HyPerLayer {
  public:
   PtwiseQuotientLayer(const char *name, PVParams *params, Communicator const *comm);
   virtual ~PtwiseQuotientLayer();

   virtual Response::Status allocateDataStructures() override;

  protected:
   PtwiseQuotientLayer();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   ActivityComponent *createActivityComponent() override;
}; // end class PtwiseQuotientLayer

} // end namespace PV

#endif /* PTWISEQUOTIENTLAYER_HPP_ */
