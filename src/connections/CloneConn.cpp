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
#include "delivery/CloneDeliveryFacade.hpp"

namespace PV {

CloneConn::CloneConn(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

CloneConn::CloneConn() {}

CloneConn::~CloneConn() {}

void CloneConn::initialize(char const *name, PVParams *params, Communicator *comm) {
   HyPerConn::initialize(name, params, comm);
}

void CloneConn::createComponentTable(char const *description) {
   HyPerConn::createComponentTable(description);
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam->getDescription(), mOriginalConnNameParam);
   }
}

BaseDelivery *CloneConn::createDeliveryObject() {
   return new CloneDeliveryFacade(name, parameters(), mCommunicator);
}

ArborList *CloneConn::createArborList() {
   return new DependentArborList(name, parameters(), mCommunicator);
}

PatchSize *CloneConn::createPatchSize() {
   return new DependentPatchSize(name, parameters(), mCommunicator);
}

SharedWeights *CloneConn::createSharedWeights() {
   return new DependentSharedWeights(name, parameters(), mCommunicator);
}

WeightsPairInterface *CloneConn::createWeightsPair() {
   return new CloneWeightsPair(name, parameters(), mCommunicator);
}

InitWeights *CloneConn::createWeightInitializer() { return nullptr; }

NormalizeBase *CloneConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *CloneConn::createWeightUpdater() { return nullptr; }

OriginalConnNameParam *CloneConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(name, parameters(), mCommunicator);
}

Response::Status CloneConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return Response::NO_ACTION;
}

} // namespace PV
