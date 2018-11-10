/*
 * ANNErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNErrorLayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/ErrScaleInternalStateBuffer.hpp"

namespace PV {

ANNErrorLayer::ANNErrorLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

ANNErrorLayer::~ANNErrorLayer() {}

int ANNErrorLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *ANNErrorLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<ErrScaleInternalStateBuffer, ANNActivityBuffer>(
         getName(), parent);
}

} // end namespace PV
