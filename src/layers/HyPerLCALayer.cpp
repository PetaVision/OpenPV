/*
 * HyPerLCALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "HyPerLCALayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/HyPerLCAInternalStateBuffer.hpp"
#include "components/TauLayerInputBuffer.hpp"

namespace PV {

HyPerLCALayer::HyPerLCALayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

HyPerLCALayer::~HyPerLCALayer() {}

int HyPerLCALayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

LayerInputBuffer *HyPerLCALayer::createLayerInput() {
   return new TauLayerInputBuffer(name, parent);
}

ActivityComponent *HyPerLCALayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerLCAInternalStateBuffer, ANNActivityBuffer>(
         getName(), parent);
}

} // end namespace PV
