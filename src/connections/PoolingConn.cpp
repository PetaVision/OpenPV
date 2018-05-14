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

PoolingConn::PoolingConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

PoolingConn::PoolingConn() {}

PoolingConn::~PoolingConn() {}

int PoolingConn::initialize(char const *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);
   return status;
}

void PoolingConn::setObserverTable() {
   BaseConnection::setObserverTable();
   mPatchSize = createPatchSize();
   if (mPatchSize) {
      addUniqueComponent(mPatchSize->getDescription(), mPatchSize);
   }
   mWeightsPair = createWeightsPair();
   if (mWeightsPair) {
      addUniqueComponent(mWeightsPair->getDescription(), mWeightsPair);
   }
}

BaseDelivery *PoolingConn::createDeliveryObject() { return new PoolingDelivery(name, parent); }

PatchSize *PoolingConn::createPatchSize() { return new PatchSize(name, parent); }

WeightsPairInterface *PoolingConn::createWeightsPair() {
   return new ImpliedWeightsPair(name, parent);
}

} // namespace PV
