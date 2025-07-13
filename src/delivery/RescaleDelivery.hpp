/*
 * RescaleDelivery.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef RESCALEDELIVERY_HPP_
#define RESCALEDELIVERY_HPP_

#include "delivery/IdentDelivery.hpp"

namespace PV {

/**
 * The delivery component for the RescaleConn class.
 * Delivers a scalar multiple of the presynaptic activity to the postsynaptic GSyn channel.
 */
class RescaleDelivery : public IdentDelivery {
  protected:
   /**
    * List of parameters needed from the RescaleDelivery class
    * @name RescaleDelivery Parameters
    * @{
    */

   /**
    * scale: presynaptic activity is multiplied by this scale factor before being added to the
    * postsynaptic input.
    */
   void ioParam_scale(ParamsIOSwitch ioSwitch);

   /** @} */
   // End of parameters needed from the RescaleConn class.

  public:
   RescaleDelivery(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual ~RescaleDelivery() {}

   virtual void deliver(float *destBuffer) override;

   virtual void deliverUnitInput(float *recvBuffer) override;

  protected:
   RescaleDelivery() {}

   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  private:
   float mScale;
}; // end class RescaleDelivery

} // end namespace PV

#endif // RESCALEDELIVERY_HPP_
