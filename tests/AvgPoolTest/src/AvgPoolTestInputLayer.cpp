#include "AvgPoolTestInputLayer.hpp"

#include "AvgPoolTestInputBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

AvgPoolTestInputLayer::AvgPoolTestInputLayer(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
}

ActivityComponent *AvgPoolTestInputLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<AvgPoolTestInputBuffer>(getName(), parent);
}

} /* namespace PV */
