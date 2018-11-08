/*
 * PoolingConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "PoolingConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/PatchSize.hpp"
#include "delivery/PoolingDelivery.hpp"

namespace PV {

PoolingConn::PoolingConn(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

PoolingConn::PoolingConn() {}

PoolingConn::~PoolingConn() {}

void PoolingConn::initialize(char const *name, PVParams *params, Communicator *comm) {
   BaseConnection::initialize(name, params, comm);
}

void PoolingConn::createComponentTable(char const *description) {
   BaseConnection::createComponentTable(description);
   mPatchSize = createPatchSize();
   if (mPatchSize) {
      addUniqueComponent(mPatchSize->getDescription(), mPatchSize);
   }
   mWeightsPair = createWeightsPair();
   if (mWeightsPair) {
      addUniqueComponent(mWeightsPair->getDescription(), mWeightsPair);
   }
}

BaseDelivery *PoolingConn::createDeliveryObject() {
   return new PoolingDelivery(name, parameters(), mCommunicator);
}

PatchSize *PoolingConn::createPatchSize() {
   return new PatchSize(name, parameters(), mCommunicator);
}

WeightsPairInterface *PoolingConn::createWeightsPair() {
   return new ImpliedWeightsPair(name, parameters(), mCommunicator);
}

} // namespace PV
