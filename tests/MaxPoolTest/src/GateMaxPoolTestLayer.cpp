#include "GateMaxPoolTestLayer.hpp"

#include "GateMaxPoolTestBuffer.hpp"
#include <components/ANNActivityBuffer.hpp>
#include <components/ActivityComponentWithInternalState.hpp>

namespace PV {

GateMaxPoolTestLayer::GateMaxPoolTestLayer(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
}

ActivityComponent *GateMaxPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<GateMaxPoolTestBuffer, ANNActivityBuffer>(
         getName(), parent);
}

} /* namespace PV */
