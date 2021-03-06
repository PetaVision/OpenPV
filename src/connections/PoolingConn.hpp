/*
 * PoolingConn.hpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#ifndef POOLINGCONN_HPP_
#define POOLINGCONN_HPP_

#include "components/ImpliedWeightsPair.hpp"
#include "components/PatchSize.hpp"
#include "connections/BaseConnection.hpp"

namespace PV {

class PoolingConn : public BaseConnection {
  public:
   PoolingConn(char const *name, PVParams *params, Communicator const *comm);

   virtual ~PoolingConn();

   // Jul 10, 2018: get-methods have been moved into the corresponding component classes.
   // For example, the old PoolingConn::getPatchSizeX() has been moved into the PatchSize class.
   // To get the PatchSizeX value from a PoolingConn conn , get the PatchSize component using
   // "PatchSize *patchsize = conn->getComponentByType<PatchSize>()" and then call
   // "patchSize->getPatchSizeX()"

  protected:
   PoolingConn();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void fillComponentTable() override;

   virtual BaseDelivery *createDeliveryObject() override;

   virtual PatchSize *createPatchSize();

   virtual WeightsPairInterface *createWeightsPair();

  protected:
   PatchSize *mPatchSize              = nullptr;
   WeightsPairInterface *mWeightsPair = nullptr;

}; // class PoolingConn

} // namespace PV

#endif // POOLINGCONN_HPP_
