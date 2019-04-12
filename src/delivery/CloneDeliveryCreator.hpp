/*
 * CloneDeliveryCreator.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef CLONEDELIVERYCREATOR_HPP_
#define CLONEDELIVERYCREATOR_HPP_

#include "delivery/HyPerDeliveryCreator.hpp"

namespace PV {

/**
 * The delivery component for CloneConns. It is exactly the same as HyPerDeliveryCreator, except
 * that it requires the WeightsPair to be a CloneWeights pair; and if updateGSynFromPostPerspective
 * is set, it synchronizes the original's and clone's postsynaptic layer margin widths.
 */
class CloneDeliveryCreator : public HyPerDeliveryCreator {
  public:
   CloneDeliveryCreator(char const *name, PVParams *params, Communicator const *comm);

   virtual ~CloneDeliveryCreator();

  protected:
   CloneDeliveryCreator();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
}; // end class CloneDeliveryCreator

} // end namespace PV

#endif // CLONEDELIVERYCREATOR_HPP_
