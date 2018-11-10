#include "MaxPoolTestLayer.hpp"

#include "MaxPoolTestBuffer.hpp"
#include <columns/HyPerCol.hpp>
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

MaxPoolTestLayer::MaxPoolTestLayer(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
}

ActivityComponent *MaxPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, MaxPoolTestBuffer>(
         getName(), parent);
}

} /* namespace PV */
