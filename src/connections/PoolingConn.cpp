/*
 * PoolingConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "PoolingConn.hpp"
#include "components/PatchSize.hpp"
#include "delivery/PoolingDelivery.hpp"

namespace PV {

PoolingConn::PoolingConn(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

PoolingConn::PoolingConn() {}

PoolingConn::~PoolingConn() {}

void PoolingConn::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   BaseConnection::initialize(params, defaults, comm);
}

void PoolingConn::fillComponentTable() {
   BaseConnection::fillComponentTable();
   mPatchSize = createPatchSize();
   if (mPatchSize) {
      addUniqueComponent(mPatchSize);
   }
   auto *sharedWeights = createSharedWeights();
   if (sharedWeights) {
      addUniqueComponent(sharedWeights);
   }
   mWeightsPair = createWeightsPair();
   if (mWeightsPair) {
      addUniqueComponent(mWeightsPair);
   }
}

BaseDelivery *PoolingConn::createDeliveryObject() {
   return new PoolingDelivery(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

PatchSize *PoolingConn::createPatchSize() {
   return new PatchSize(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

SharedWeights *PoolingConn::createSharedWeights() {
   return new SharedWeights(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

WeightsPairInterface *PoolingConn::createWeightsPair() {
   return new ImpliedWeightsPair(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // namespace PV
