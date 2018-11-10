/*
 * SigmoidLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "SigmoidLayer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/SigmoidActivityBuffer.hpp"

// SigmoidLayer can be used to implement Sigmoid junctions
namespace PV {
SigmoidLayer::SigmoidLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

SigmoidLayer::SigmoidLayer() {}

SigmoidLayer::~SigmoidLayer() {}

int SigmoidLayer::initialize(const char *name, HyPerCol *hc) {
   int status_init = CloneVLayer::initialize(name, hc);
   return status_init;
}

ActivityComponent *SigmoidLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<CloneInternalStateBuffer, SigmoidActivityBuffer>(
         getName(), parent);
}

} // end namespace PV
