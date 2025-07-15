#include "AlwaysFailsLayer.hpp"

namespace PV {

AlwaysFailsLayer::AlwaysFailsLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

AlwaysFailsLayer::AlwaysFailsLayer() {}

AlwaysFailsLayer::~AlwaysFailsLayer() {}

void AlwaysFailsLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

Response::Status AlwaysFailsLayer::checkUpdateState(double simTime, double deltaTime) {
   // The params file should be run with the -n flag, which causes HyPerCol::run() to exit before
   // entering the advanceTime loop.
   // Therefore LayerUpdateState should never be called.
   Fatal() << getDescription()
           << ": needUpdate was called, and should never be called during DryRunFlagTest.\n";
   return Response::SUCCESS;
}

} // end namespace PV
