/*
 * ISTALayer.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "ISTALayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/ISTAInternalStateBuffer.hpp"
#include "components/TauLayerInputBuffer.hpp"

namespace PV {

ISTALayer::ISTALayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

ISTALayer::~ISTALayer() {}

int ISTALayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

LayerInputBuffer *ISTALayer::createLayerInput() { return new TauLayerInputBuffer(name, parent); }

ActivityComponent *ISTALayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<ISTAInternalStateBuffer, ANNActivityBuffer>(
         getName(), parent);
}

} // end namespace PV
