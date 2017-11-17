/*
 * IdentDelivery.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef IDENTDELIVERY_HPP_
#define IDENTDELIVERY_HPP_

#include "components/BaseDelivery.hpp"

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

   virtual void deliver(Weights *weights, Weights *postWeights) override;

  protected:
   IdentDelivery() {}

   int initialize(char const *name, HyPerCol *hc);

   void checkDimensions(PVLayerLoc const &preLoc, PVLayerLoc const &postLoc) const;
}; // end class IdentDelivery

} // end namespace PV

#endif // IDENTDELIVERY_HPP_
