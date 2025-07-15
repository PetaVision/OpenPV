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

BackgroundLayer::BackgroundLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

BackgroundLayer::~BackgroundLayer() {}

void BackgroundLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
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
         mParamsIO, mCommunicator);
}

OriginalLayerNameParam *BackgroundLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO, mCommunicator);
}

} // end namespace PV
