/*
 * RescaleLayer.cpp
 */

#include "RescaleLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/RescaleActivityBuffer.hpp"

namespace PV {

RescaleLayer::RescaleLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

RescaleLayer::RescaleLayer() {}

RescaleLayer::~RescaleLayer() {}

int RescaleLayer::initialize(const char *name, HyPerCol *hc) {
   int status = CloneVLayer::initialize(name, hc);
   return status;
}

ActivityComponent *RescaleLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<RescaleActivityBuffer>(getName(), parent);
}

} // end namespace PV
