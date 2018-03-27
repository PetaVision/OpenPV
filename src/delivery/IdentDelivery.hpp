/*
 * IdentDelivery.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef IDENTDELIVERY_HPP_
#define IDENTDELIVERY_HPP_

#include "BaseDelivery.hpp"
#include "components/SingleArbor.hpp"

namespace PV {

/**
 * The delivery component for the IdentConn class.
 * Delivers the presynaptic activity unaltered to the postsynaptic GSyn channel.
 */
class IdentDelivery : public BaseDelivery {
  protected:
   /**
    * List of parameters needed from the IdentDelivery class
    * @name IdentDelivery Parameters
    * @{
    */

   /**
    * @brief IdentDeliver does not use the GPU. It is an error to set receiveGpu to true.
    */
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag) override;
   /** @} */ // End of list of IdentDelivery parameters.

  public:
   IdentDelivery(char const *name, HyPerCol *hc);

   virtual ~IdentDelivery() {}

   virtual void deliver() override;

   virtual void deliverUnitInput(float *recvBuffer) override;

   virtual bool isAllInputReady() override;

  protected:
   IdentDelivery() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   void checkPreAndPostDimensions();

  protected:
   SingleArbor *mSingleArbor = nullptr;

}; // end class IdentDelivery

} // end namespace PV

#endif // IDENTDELIVERY_HPP_
