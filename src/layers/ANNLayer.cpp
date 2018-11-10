/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

ANNLayer::ANNLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

ANNLayer::~ANNLayer() {}

int ANNLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *ANNLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, ANNActivityBuffer>(
         getName(), parent);
}

} // end namespace PV
