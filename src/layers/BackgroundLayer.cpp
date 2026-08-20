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

BackgroundLayer::BackgroundLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

BackgroundLayer::~BackgroundLayer() {}

void BackgroundLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseLayer::initialize(name, params, comm);
}

void BackgroundLayer::fillComponentTable() {
   BaseLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

ActivityComponent *BackgroundLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<BackgroundActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

OriginalLayerNameParam *BackgroundLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(getName(), parameters(), mCommunicator);
}

} // end namespace PV
