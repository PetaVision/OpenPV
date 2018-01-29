/*
 * CloneDeliveryFacade.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef CLONEDELIVERYFACADE_HPP_
#define CLONEDELIVERYFACADE_HPP_

#include "delivery/HyPerDeliveryFacade.hpp"

namespace PV {

/**
 * The delivery component for CloneConns. It is exactly the same as HyPerDeliveryFacade, except
 * that it requires the WeightsPair to be a CloneWeights pair; and if updateGSynFromPostPerspective
 * is set, it synchronizes the original's and clone's postsynaptic layer margin widths.
 */
class CloneDeliveryFacade : public HyPerDeliveryFacade {
  public:
   CloneDeliveryFacade(char const *name, HyPerCol *hc);

   virtual ~CloneDeliveryFacade();

  protected:
   CloneDeliveryFacade();

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
}; // end class CloneDeliveryFacade

} // end namespace PV

#endif // CLONEDELIVERYFACADE_HPP_
