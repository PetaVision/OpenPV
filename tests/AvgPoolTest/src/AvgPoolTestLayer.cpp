#include "AvgPoolTestLayer.hpp"

#include "AvgPoolTestBuffer.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

AvgPoolTestLayer::AvgPoolTestLayer(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
}

ActivityComponent *AvgPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, AvgPoolTestBuffer>(
         getName(), parent);
}

} /* namespace PV */
