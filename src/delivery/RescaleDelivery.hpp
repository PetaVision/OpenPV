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
   void ioParam_scale(enum ParamsIOFlag ioFlag);

   /** @} */
   // End of parameters needed from the RescaleConn class.

  public:
   RescaleDelivery(char const *name, HyPerCol *hc);

   virtual ~RescaleDelivery() {}

   virtual void deliver() override;

   virtual void deliverUnitInput(float *recvBuffer) override;

  protected:
   RescaleDelivery() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  private:
   float mScale;
}; // end class RescaleDelivery

} // end namespace PV

#endif // RESCALEDELIVERY_HPP_
