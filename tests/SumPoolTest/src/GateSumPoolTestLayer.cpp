#include "GateSumPoolTestLayer.hpp"

#include "GateSumPoolTestBuffer.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

GateSumPoolTestLayer::GateSumPoolTestLayer(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
}

ActivityComponent *GateSumPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, GateSumPoolTestBuffer>(
         getName(), parent);
}

} /* namespace PV */
