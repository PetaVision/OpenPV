/*
 * PoolingConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "PoolingConn.hpp"
#include "components/PatchSize.hpp"
#include "components/SpecifiedSharedWeights.hpp"
#include "delivery/PoolingDelivery.hpp"

namespace PV {

PoolingConn::PoolingConn(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

PoolingConn::PoolingConn() {}

PoolingConn::~PoolingConn() {}

void PoolingConn::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   BaseConnection::initialize(paramsIO, comm);
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
   return new PoolingDelivery(mParamsIO, mCommunicator);
}

PatchSize *PoolingConn::createPatchSize() {
   return new PatchSize(mParamsIO, mCommunicator);
}

SharedWeights *PoolingConn::createSharedWeights() {
   return new SpecifiedSharedWeights<false>(mParamsIO, mCommunicator);
}

WeightsPairInterface *PoolingConn::createWeightsPair() {
   return new ImpliedWeightsPair(mParamsIO, mCommunicator);
}

} // namespace PV
