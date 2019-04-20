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
   HyPerLayer::initialize(name, params, comm);
}

void BinningLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

LayerInputBuffer *BinningLayer::createLayerInput() { return nullptr; }

ActivityComponent *BinningLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<BinningActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

OriginalLayerNameParam *BinningLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parameters(), mCommunicator);
}

} /* namespace PV */
