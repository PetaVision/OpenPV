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

/**
 * The output V is the pointwise product of GSynExc and GSynInh
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 * The activity buffer is an ANNActivityBuffer.
 */
class PtwiseProductLayer : public HyPerLayer {
  public:
   PtwiseProductLayer(const char *name, PVParams *params, Communicator *comm);
   virtual ~PtwiseProductLayer();

   virtual Response::Status allocateDataStructures() override;

  protected:
   PtwiseProductLayer();
   void initialize(const char *name, PVParams *params, Communicator *comm);
   ActivityComponent *createActivityComponent() override;
}; // end class PtwiseProductLayer

} // end namespace PV

#endif /* PTWISEPRODUCTLAYER_HPP_ */
