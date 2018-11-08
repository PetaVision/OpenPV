#include "GateSumPoolTestLayer.hpp"

#include "GateSumPoolTestBuffer.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

GateSumPoolTestLayer::GateSumPoolTestLayer(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *GateSumPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, GateSumPoolTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
