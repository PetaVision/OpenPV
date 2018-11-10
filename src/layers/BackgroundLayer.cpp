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

BackgroundLayer::BackgroundLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

BackgroundLayer::~BackgroundLayer() {}

int BackgroundLayer::initialize(const char *name, HyPerCol *hc) {
   int status_init = HyPerLayer::initialize(name, hc);

   return status_init;
}

void BackgroundLayer::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

LayerInputBuffer *BackgroundLayer::createLayerInput() { return nullptr; }

ActivityComponent *BackgroundLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<BackgroundActivityBuffer>(getName(), parent);
}

OriginalLayerNameParam *BackgroundLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

} // end namespace PV
