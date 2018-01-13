/*
 * PostsynapticPerspectiveStochasticDelivery.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef POSTSYNAPTICPERSPECTIVESTOCHASTICDELIVERY_HPP_
#define POSTSYNAPTICPERSPECTIVESTOCHASTICDELIVERY_HPP_

#include "delivery/HyPerDelivery.hpp"

namespace PV {

/**
 * The delivery class for HyPerConns using the postsynaptic perspective on the CPU,
 * with accumulate type "convolve".
 */
class PostsynapticPerspectiveStochasticDelivery : public HyPerDelivery {
  protected:
   /**
    * List of parameters needed from the PostsynapticPerspectiveStochasticDelivery class
    * @name PostsynapticPerspectiveStochasticDelivery Parameters
    * @{
    */

   /**
    * @brief receiveGpu: PostsynapticPerspectiveStochasticDelivery always sets receiveGpu to false.
    * The receiveGpu=true case is handled by the PostsynapticPerspectiveGPUDelivery class.
    */
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag) override;
   /** @} */ // End of list of BaseDelivery parameters.

  public:
   PostsynapticPerspectiveStochasticDelivery(char const *name, HyPerCol *hc);

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
    *
    * The postWeights argument is not used.
    */
   virtual void deliver() override;

   virtual void deliverUnitInput(float *recvBuffer) override;

  protected:
   PostsynapticPerspectiveStochasticDelivery();

   int initialize(char const *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual int allocateDataStructures() override;

   void allocateThreadGSyn();

   // Data members
  protected:
   std::vector<std::vector<float>> mThreadGSyn;
   Random *mRandState = nullptr;

}; // end class PostsynapticPerspectiveStochasticDelivery

} // end namespace PV

#endif // POSTSYNAPTICPERSPECTIVESTOCHASTICDELIVERY_HPP_
