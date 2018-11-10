/*
 *  Created on: Jan 15, 2014
 *      Author: Sheng Lundquist
 */

#include "BinningLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/BinningActivityBuffer.hpp"

namespace PV {
BinningLayer::BinningLayer() {}

BinningLayer::BinningLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

BinningLayer::~BinningLayer() {}

int BinningLayer::initialize(const char *name, HyPerCol *hc) {
   int status_init = HyPerLayer::initialize(name, hc);

   return status_init;
}

void BinningLayer::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

LayerInputBuffer *BinningLayer::createLayerInput() { return nullptr; }

ActivityComponent *BinningLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<BinningActivityBuffer>(getName(), parent);
}

OriginalLayerNameParam *BinningLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

} /* namespace PV */
