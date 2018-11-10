/*
 * GapLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "GapLayer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/GapActivityBuffer.hpp"

namespace PV {

GapLayer::GapLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

GapLayer::~GapLayer() {}

int GapLayer::initialize(const char *name, HyPerCol *hc) {
   int status = CloneVLayer::initialize(name, hc);
   return status;
}

ActivityComponent *GapLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<CloneInternalStateBuffer, GapActivityBuffer>(
         getName(), parent);
}

} // end namespace PV
