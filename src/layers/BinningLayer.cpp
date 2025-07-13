/*
 *  Created on: Jan 15, 2014
 *      Author: Sheng Lundquist
 */

#include "BinningLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/BinningActivityBuffer.hpp"

namespace PV {
BinningLayer::BinningLayer() {}

BinningLayer::BinningLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

BinningLayer::~BinningLayer() {}

void BinningLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
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
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

OriginalLayerNameParam *BinningLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} /* namespace PV */
