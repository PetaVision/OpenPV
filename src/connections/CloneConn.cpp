/* CloneConn.cpp
 *
 * Created on: May 23, 2011
 *     Author: peteschultz
 */

#include "CloneConn.hpp"
#include "components/CloneWeightsPair.hpp"
#include "components/DependentArborList.hpp"
#include "components/DependentPatchSize.hpp"
#include "components/DependentSharedWeights.hpp"
#include "delivery/CloneDeliveryCreator.hpp"

namespace PV {

CloneConn::CloneConn(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

CloneConn::CloneConn() {}

CloneConn::~CloneConn() {}

void CloneConn::initialize(char const *name, PVParams *params, Communicator const *comm) {
   HyPerConn::initialize(name, params, comm);
}

void CloneConn::fillComponentTable() {
   HyPerConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

BaseDelivery *CloneConn::createDeliveryObject() {
   auto *deliveryCreator = new CloneDeliveryCreator(getName(), parameters(), mCommunicator);
   addUniqueComponent(deliveryCreator);
   return deliveryCreator->create();
}

ArborList *CloneConn::createArborList() {
   return new DependentArborList(getName(), parameters(), mCommunicator);
}

PatchSize *CloneConn::createPatchSize() {
   return new DependentPatchSize(getName(), parameters(), mCommunicator);
}

SharedWeights *CloneConn::createSharedWeights() {
   return new DependentSharedWeights(getName(), parameters(), mCommunicator);
}

WeightsPairInterface *CloneConn::createWeightsPair() {
   return new CloneWeightsPair(getName(), parameters(), mCommunicator);
}

InitWeights *CloneConn::createWeightInitializer() { return nullptr; }

NormalizeBase *CloneConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *CloneConn::createWeightUpdater() { return nullptr; }

OriginalConnNameParam *CloneConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(getName(), parameters(), mCommunicator);
}

Response::Status CloneConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return Response::NO_ACTION;
}

} // namespace PV
