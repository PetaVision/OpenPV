/*
 * TopDownDelivery.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef TOPDOWNDELIVERY_HPP_
#define TOPDOWNDELIVERY_HPP_

#include "delivery/IdentDelivery.hpp"

namespace PV {

/**
 * The delivery component for the RescaleConn class.
 * Delivers a scalar multiple of the presynaptic activity to the postsynaptic GSyn channel.
 */
class TopDownDelivery : public IdentDelivery {
  protected:
   /**
    * List of parameters needed from the TopDownDelivery class
    * @name TopDownDelivery Parameters
    * @{
    */

   void ioParam_displayPeriod(enum ParamsIOFlag ioFlag);

   void ioParam_zeroRatio(enum ParamsIOFlag ioFlag);
   /** @} */
   // End of parameters needed from the RescaleConn class.

  public:
   TopDownDelivery(char const *name, HyPerCol *hc);

   virtual ~TopDownDelivery() {}

   virtual void deliver() override;

   virtual void deliverUnitInput(float *recvBuffer) override;

  protected:
   TopDownDelivery() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  private:
   int mDeliverCount  = 0;
   int mDisplayPeriod = -1;
   float mZeroRatio   = 0.4f;

}; // end class TopDownDelivery

} // end namespace PV

#endif // TOPDOWNDELIVERY_HPP_
