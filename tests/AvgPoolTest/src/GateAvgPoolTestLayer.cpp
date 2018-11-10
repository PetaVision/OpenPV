#include "GateAvgPoolTestLayer.hpp"

#include "GateAvgPoolTestBuffer.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

GateAvgPoolTestLayer::GateAvgPoolTestLayer(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
}

ActivityComponent *GateAvgPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, GateAvgPoolTestBuffer>(
         getName(), parent);
}

} /* namespace PV */
