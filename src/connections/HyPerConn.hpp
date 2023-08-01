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
#include "delivery/BaseDelivery.hpp"
#include "normalizers/NormalizeBase.hpp"
#include "weightinit/InitWeights.hpp"
#include "weightupdaters/BaseWeightUpdater.hpp"

namespace PV {

class HyPerConn : public BaseConnection {
  public:
   HyPerConn(char const *name, PVParams *params, Communicator const *comm);

   virtual ~HyPerConn();

  protected:
   HyPerConn();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void initMessageActionMap() override;

   virtual void fillComponentTable() override;

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

  private:
   void warnIfBroadcastWithShared();

  protected:
   Timer *mUpdateTimer = nullptr;

}; // class HyPerConn

} // namespace PV

#endif // HYPERCONN_HPP_
