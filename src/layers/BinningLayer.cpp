/*
 *  Created on: Jan 15, 2014
 *      Author: Sheng Lundquist
 */

#include "BinningLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/BinningActivityBuffer.hpp"

namespace PV {
BinningLayer::BinningLayer() {}

BinningLayer::BinningLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

BinningLayer::~BinningLayer() {}

void BinningLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
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
         mParamsIO, mCommunicator);
}

OriginalLayerNameParam *BinningLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO, mCommunicator);
}

} /* namespace PV */
