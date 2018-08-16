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
 * The delivery component for the IdentConn class.
 * Delivers the presynaptic activity unaltered to the postsynaptic GSyn channel.
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
   WTADelivery(char const *name, HyPerCol *hc);

   virtual ~WTADelivery() {}

   virtual void deliver() override;

   virtual void deliverUnitInput(float *recvBuffer) override;

   virtual bool isAllInputReady() override;

  protected:
   WTADelivery() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   void checkPreAndPostDimensions();

  protected:
   int mDelay = 0;

}; // end class WTADelivery

} // end namespace PV

#endif // WTADELIVERY_HPP_
