#include "SumPoolTestLayer.hpp"

#include "SumPoolTestBuffer.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

SumPoolTestLayer::SumPoolTestLayer(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
}

ActivityComponent *SumPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, SumPoolTestBuffer>(
         getName(), parent);
}

} /* namespace PV */
