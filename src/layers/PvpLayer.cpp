#include "PvpLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/PvpActivityBuffer.hpp"

namespace PV {

PvpLayer::PvpLayer(char const *name, HyPerCol *hc) { initialize(name, hc); }

PvpLayer::~PvpLayer() {}

int PvpLayer::initialize(char const *name, HyPerCol *hc) {
   return InputLayer::initialize(name, hc);
}

ActivityComponent *PvpLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<PvpActivityBuffer>(getName(), parent);
}

} // end namespace PV
