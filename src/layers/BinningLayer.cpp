/*
 *  Created on: Jan 15, 2014
 *      Author: Sheng Lundquist
 */

#include "BinningLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/BinningActivityBuffer.hpp"

namespace PV {
BinningLayer::BinningLayer() {}

BinningLayer::BinningLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

BinningLayer::~BinningLayer() {}

void BinningLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseLayer::initialize(name, params, comm);
}

void BinningLayer::fillComponentTable() {
   BaseLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

ActivityComponent *BinningLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<BinningActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

OriginalLayerNameParam *BinningLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(getName(), parameters(), mCommunicator);
}

} /* namespace PV */
