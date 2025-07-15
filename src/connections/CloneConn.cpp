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

CloneConn::CloneConn(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

CloneConn::CloneConn() {}

CloneConn::~CloneConn() {}

void CloneConn::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerConn::initialize(paramsIO, comm);
}

void CloneConn::fillComponentTable() {
   HyPerConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

BaseDelivery *CloneConn::createDeliveryObject() {
   auto *deliveryCreator = new CloneDeliveryCreator(mParamsIO, mCommunicator);
   addUniqueComponent(deliveryCreator);
   return deliveryCreator->create();
}

ArborList *CloneConn::createArborList() {
   return new DependentArborList(mParamsIO, mCommunicator);
}

PatchSize *CloneConn::createPatchSize() {
   return new DependentPatchSize(mParamsIO, mCommunicator);
}

SharedWeights *CloneConn::createSharedWeights() {
   return new DependentSharedWeights(mParamsIO, mCommunicator);
}

WeightsPairInterface *CloneConn::createWeightsPair() {
   return new CloneWeightsPair(mParamsIO, mCommunicator);
}

InitWeights *CloneConn::createWeightInitializer() { return nullptr; }

NormalizeBase *CloneConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *CloneConn::createWeightUpdater() { return nullptr; }

OriginalConnNameParam *CloneConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(mParamsIO, mCommunicator);
}

Response::Status CloneConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return Response::NO_ACTION;
}

} // namespace PV
