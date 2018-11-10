/*
 * LIF.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 */

#include "LIF.hpp"
#include "components/LIFActivityComponent.hpp"
#include "components/LIFLayerInputBuffer.hpp"

namespace PV {

LIF::LIF(const char *name, HyPerCol *hc) { initialize(name, hc); }

LIF::LIF() {}

LIF::~LIF() {}

int LIF::initialize(const char *name, HyPerCol *hc) { return HyPerLayer::initialize(name, hc); }

LayerInputBuffer *LIF::createLayerInput() { return new LIFLayerInputBuffer(getName(), parent); }

ActivityComponent *LIF::createActivityComponent() {
   return new LIFActivityComponent(getName(), parent);
}

} // end namespace PV
