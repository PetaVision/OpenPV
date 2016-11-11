#include "AlwaysFailsLayer.hpp"

namespace PV {

AlwaysFailsLayer::AlwaysFailsLayer(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

AlwaysFailsLayer::AlwaysFailsLayer() { initialize_base(); }

AlwaysFailsLayer::~AlwaysFailsLayer() {}

int AlwaysFailsLayer::initialize_base() { return PV_SUCCESS; }

int AlwaysFailsLayer::initialize(char const *name, HyPerCol *hc) {
   return HyPerLayer::initialize(name, hc);
}

bool AlwaysFailsLayer::needUpdate(double simTime, double dt) {
   // The params file should be run with the -n flag, which causes HyPerCol::run() to exit before
   // entering the advanceTime loop.
   // Therefore neither updateState, nor updateStateGpu, should ever be called.
   Fatal() << getDescription()
           << ": needUpdate was called, and should never be called during DryRunFlagTest.\n";
   return false;
}

} // end namespace PV