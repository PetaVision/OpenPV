/*
 * PoolingConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "PoolingConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/NoCheckpointConnectionData.hpp"
#include "delivery/PoolingDelivery.hpp"

namespace PV {

PoolingConn::PoolingConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

PoolingConn::PoolingConn() {}

PoolingConn::~PoolingConn() {}

int PoolingConn::initialize(char const *name, HyPerCol *hc) {
   int status = BaseConnection::initialize(name, hc);
   return status;
}

void PoolingConn::defineComponents() {
   BaseConnection::defineComponents();
   mWeightsPair = createWeightsPair();
   if (mWeightsPair) {
      addObserver(mWeightsPair);
   }
}

ImpliedWeightsPair *PoolingConn::createWeightsPair() {
   return new ImpliedWeightsPair(name, parent);
}

ConnectionData *PoolingConn::createConnectionData() {
   return new NoCheckpointConnectionData(name, parent);
}

BaseDelivery *PoolingConn::createDeliveryObject() { return new PoolingDelivery(name, parent); }

} // namespace PV
