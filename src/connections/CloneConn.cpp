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

CloneConn::CloneConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

CloneConn::CloneConn() {}

CloneConn::~CloneConn() {}

void CloneConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerConn::initialize(params, defaults, comm);
}

void CloneConn::fillComponentTable() {
   HyPerConn::fillComponentTable();
   mOriginalConnNameParam = createOriginalConnNameParam();
   if (mOriginalConnNameParam) {
      addUniqueComponent(mOriginalConnNameParam);
   }
}

BaseDelivery *CloneConn::createDeliveryObject() {
   auto *deliveryCreator = new CloneDeliveryCreator(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
   addUniqueComponent(deliveryCreator);
   return deliveryCreator->create();
}

ArborList *CloneConn::createArborList() {
   return new DependentArborList(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

PatchSize *CloneConn::createPatchSize() {
   return new DependentPatchSize(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

SharedWeights *CloneConn::createSharedWeights() {
   return new DependentSharedWeights(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

WeightsPairInterface *CloneConn::createWeightsPair() {
   return new CloneWeightsPair(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

InitWeights *CloneConn::createWeightInitializer() { return nullptr; }

NormalizeBase *CloneConn::createWeightNormalizer() { return nullptr; }

BaseWeightUpdater *CloneConn::createWeightUpdater() { return nullptr; }

OriginalConnNameParam *CloneConn::createOriginalConnNameParam() {
   return new OriginalConnNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

Response::Status CloneConn::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return Response::NO_ACTION;
}

} // namespace PV
