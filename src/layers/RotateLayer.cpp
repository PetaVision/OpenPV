#include "RotateLayer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/RotateActivityBuffer.hpp"

namespace PV {

RotateLayer::RotateLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void RotateLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *RotateLayer::createActivityComponent() {
   return new HyPerActivityComponent<
         GSynAccumulator,
         HyPerInternalStateBuffer,
         RotateActivityBuffer>(getName(), parameters(), mCommunicator);
}

} // namespace PV
