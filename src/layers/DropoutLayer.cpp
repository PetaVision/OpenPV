/*
 * DropoutLayer.cpp
 */

#include "DropoutLayer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/DropoutActivityBuffer.hpp"
#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

DropoutLayer::DropoutLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

DropoutLayer::~DropoutLayer() {}

int DropoutLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

ActivityComponent *DropoutLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, DropoutActivityBuffer>(
         getName(), parent);
}

} // end namespace PV
