#include "AlwaysFailsLayer.hpp"

namespace PV {

AlwaysFailsLayer::AlwaysFailsLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

AlwaysFailsLayer::AlwaysFailsLayer() {}

AlwaysFailsLayer::~AlwaysFailsLayer() {}

void AlwaysFailsLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
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
