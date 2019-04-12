/*
 * PostsynapticPerspectiveStochasticDelivery.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef POSTSYNAPTICPERSPECTIVESTOCHASTICDELIVERY_HPP_
#define POSTSYNAPTICPERSPECTIVESTOCHASTICDELIVERY_HPP_

#include "delivery/HyPerDelivery.hpp"

#include "columns/Random.hpp"

namespace PV {

/**
 * The delivery class for HyPerConns using the postsynaptic perspective on the CPU,
 * with accumulate type "convolve".
 */
class PostsynapticPerspectiveStochasticDelivery : public HyPerDelivery {
  public:
   PostsynapticPerspectiveStochasticDelivery(
         char const *name,
         PVParams *params,
         Communicator const *comm);

   virtual ~PostsynapticPerspectiveStochasticDelivery();

   /**
    * The method that delivers presynaptic activity to the given postsynaptic channel.
    * It loops over presynaptic neurons, skipping over any whose activity is zero
    * (to take advantage of sparsity). Each neuron then modifies the region of the post channel
    * that the weights argument specifies for that pre-synaptic neuron.
    *
    * If OpenMP is used, we parallelize over the presynaptic neuron. To avoid the
    * possibility of collisions where more than one pre-neuron writes to the
    * same post-neuron, we internally allocate multiple buffers the size of the post channel,
    * and accumulate them at the end.
    */
   virtual void deliver(float *destBuffer) override;

   virtual void deliverUnitInput(float *recvBuffer) override;

  protected:
   PostsynapticPerspectiveStochasticDelivery();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   // Data members
  protected:
   Random *mRandState = nullptr;

}; // end class PostsynapticPerspectiveStochasticDelivery

} // end namespace PV

#endif // POSTSYNAPTICPERSPECTIVESTOCHASTICDELIVERY_HPP_
