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

PoolingConn::PoolingConn(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

PoolingConn::PoolingConn() {}

PoolingConn::~PoolingConn() {}

void PoolingConn::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseConnection::initialize(name, params, comm);
}

void PoolingConn::fillComponentTable() {
   BaseConnection::fillComponentTable();
   mPatchSize = createPatchSize();
   if (mPatchSize) {
      addUniqueComponent(mPatchSize);
   }
   mWeightsPair = createWeightsPair();
   if (mWeightsPair) {
      addUniqueComponent(mWeightsPair);
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
