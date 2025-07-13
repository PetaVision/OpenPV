/*
 * BackgroundLayer.cpp
 *
 *  Created on: 4/16/15
 *  slundquist
 */

#include "BackgroundLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/BackgroundActivityBuffer.hpp"

namespace PV {
BackgroundLayer::BackgroundLayer() {}

BackgroundLayer::BackgroundLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

BackgroundLayer::~BackgroundLayer() {}

void BackgroundLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

void BackgroundLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

LayerInputBuffer *BackgroundLayer::createLayerInput() { return nullptr; }

ActivityComponent *BackgroundLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<BackgroundActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

OriginalLayerNameParam *BackgroundLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
