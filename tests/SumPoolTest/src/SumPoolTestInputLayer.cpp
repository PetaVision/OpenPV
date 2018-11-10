#include "SumPoolTestInputLayer.hpp"

#include "SumPoolTestInputBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

SumPoolTestInputLayer::SumPoolTestInputLayer(const char *name, HyPerCol *hc) {
   HyPerLayer::initialize(name, hc);
}

ActivityComponent *SumPoolTestInputLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<SumPoolTestInputBuffer>(getName(), parent);
}

} /* namespace PV */
