/*
 * WTADelivery.hpp
 *
 *  Created on: Aug 15, 2018
 *      Author: Pete Schultz
 */

#ifndef WTADELIVERY_HPP_
#define WTADELIVERY_HPP_

#include "BaseDelivery.hpp"
#include "components/SingleArbor.hpp"

namespace PV {

/**
 * The delivery component for the WTAConn class.
 * The pre- and post-synaptic layers must have the same nx and ny.
 * The postsynaptic layer must have nf=1.
 * The deliver method determines for each location (x,y) the maximum
 * of the presynaptic activity at (x,y,f) over f=0,...,(nfPre-1). It then
 * increments the postsynaptic GSyn at (x,y) by that maximum value.
 * A connection that uses a WTADelivery component must use a SingleArbor
 * as its ArborList component.
 */
class WTADelivery : public BaseDelivery {
  protected:
   /**
    * List of parameters needed from the WTADelivery class
    * @name WTADelivery Parameters
    * @{
    */

   /**
    * @brief WTADelivery does not use the GPU. It is an error to set receiveGpu to true.
    */
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag) override;
   /** @} */ // End of list of WTADelivery parameters.

  public:
   WTADelivery(char const *name, PVParams *params, Communicator *comm);

   virtual ~WTADelivery() {}

   virtual void deliver(float *destBuffer) override;

   virtual void deliverUnitInput(float *recvBuffer) override;

   virtual bool isAllInputReady() const override;

  protected:
   WTADelivery() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   /**
    * Verifies that pre- and post-synaptic nx values are equal,
    * that pre-and post-synaptic ny values are equal, and that postsynaptic nf=1.
    * It is a fatal error if these conditions are not met.
    */
   void checkPreAndPostDimensions();

  protected:
   int mDelay = 0;

}; // end class WTADelivery

} // end namespace PV

#endif // WTADELIVERY_HPP_
