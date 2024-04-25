#include "ScaleXLayer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/ScaleXActivityBuffer.hpp"

namespace PV {

ScaleXLayer::ScaleXLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void ScaleXLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *ScaleXLayer::createActivityComponent() {
   return new HyPerActivityComponent<
         GSynAccumulator,
         HyPerInternalStateBuffer,
         ScaleXActivityBuffer>(getName(), parameters(), mCommunicator);
}

} // namespace PV
