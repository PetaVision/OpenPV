
/*
 * MaskLayer.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#include "MaskLayer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/MaskActivityBuffer.hpp"

namespace PV {

MaskLayer::MaskLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

MaskLayer::~MaskLayer() {}

int MaskLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *MaskLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, MaskActivityBuffer>(
         getName(), parent);
}

} // end namespace PV
