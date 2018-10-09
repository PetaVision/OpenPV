/*
 * HyPerConn.hpp
 *
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "components/ArborList.hpp"
#include "components/ConnectionData.hpp"
#include "components/SharedWeights.hpp"
#include "components/WeightsPair.hpp"
#include "connections/BaseConnection.hpp"
#include "normalizers/NormalizeBase.hpp"
#include "weightinit/InitWeights.hpp"
#include "weightupdaters/BaseWeightUpdater.hpp"

namespace PV {

class HyPerCol;

class HyPerConn : public BaseConnection {
  public:
   HyPerConn(char const *name, HyPerCol *hc);

   virtual ~HyPerConn();

   // Jul 10, 2018: get-methods have been moved into the corresponding component classes.
   // For example, the old HyPerConn::getPatchSizeX() has been moved into the PatchSize class.
   // To get the PatchSizeX value from a HyPerConn conn , get the PatchSize component using
   // "PatchSize *patchsize = conn->getComponentByType<PatchSize>()" and then call
   // "patchSize->getPatchSizeX()"

  protected:
   HyPerConn();

   int initialize(char const *name, HyPerCol *hc);

   virtual void initMessageActionMap() override;

   virtual void createComponentTable(char const *description) override;

   virtual BaseDelivery *createDeliveryObject() override;
   virtual ArborList *createArborList();
   virtual PatchSize *createPatchSize();
   virtual SharedWeights *createSharedWeights();
   virtual WeightsPairInterface *createWeightsPair();
   virtual InitWeights *createWeightInitializer();
   virtual NormalizeBase *createWeightNormalizer();
   virtual BaseWeightUpdater *createWeightUpdater();

   Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   Response::Status respondConnectionUpdate(std::shared_ptr<ConnectionUpdateMessage const> message);

   Response::Status
   respondConnectionNormalize(std::shared_ptr<ConnectionNormalizeMessage const> message);

   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

  protected:
   //   ArborList *mArborList              = nullptr;
   //   PatchSize *mPatchSize              = nullptr;
   //   SharedWeights *mSharedWeights      = nullptr;
   //   WeightsPairInterface *mWeightsPair = nullptr;
   //   InitWeights *mWeightInitializer    = nullptr;
   //   NormalizeBase *mWeightNormalizer   = nullptr;
   //   BaseWeightUpdater *mWeightUpdater  = nullptr;

   Timer *mUpdateTimer = nullptr;

}; // class HyPerConn

} // namespace PV

#endif // HYPERCONN_HPP_
