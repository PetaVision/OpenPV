/*
 * MomentumLCALayer.cpp
 *
 *  Created on: Mar 15, 2016
 *      Author: slundquist
 */

#include "MomentumLCALayer.hpp"
#include "components/MomentumLCAActivityComponent.hpp"
#include "components/TauLayerInputBuffer.hpp"

namespace PV {

MomentumLCALayer::MomentumLCALayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

MomentumLCALayer::~MomentumLCALayer() {}

int MomentumLCALayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *MomentumLCALayer::createActivityComponent() {
   return new MomentumLCAActivityComponent(getName(), parent);
}

} // end namespace PV
