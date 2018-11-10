/*
 * CloneVLayer.cpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#include "CloneVLayer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/HyPerActivityBuffer.hpp"

namespace PV {

CloneVLayer::CloneVLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

CloneVLayer::CloneVLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

CloneVLayer::~CloneVLayer() {}

int CloneVLayer::initialize_base() { return PV_SUCCESS; }

int CloneVLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void CloneVLayer::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

OriginalLayerNameParam *CloneVLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

LayerInputBuffer *CloneVLayer::createLayerInput() { return nullptr; }

ActivityComponent *CloneVLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<CloneInternalStateBuffer, HyPerActivityBuffer>(
         name, parent);
}

} /* namespace PV */
