#include "AlwaysFailsLayer.hpp"

namespace PV {

AlwaysFailsLayer::AlwaysFailsLayer(char const *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

AlwaysFailsLayer::AlwaysFailsLayer() { initialize_base(); }

AlwaysFailsLayer::~AlwaysFailsLayer() {}

int AlwaysFailsLayer::initialize_base() { return PV_SUCCESS; }

void AlwaysFailsLayer::initialize(char const *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

Response::Status AlwaysFailsLayer::checkUpdateState(double simTime, double deltaTime) {
   // The params file should be run with the -n flag, which causes HyPerCol::run() to exit before
   // entering the advanceTime loop.
   // Therefore LayerUpdateState should ever be called.
   Fatal() << getDescription()
           << ": needUpdate was called, and should never be called during DryRunFlagTest.\n";
   return Response::SUCCESS;
}

} // end namespace PV
