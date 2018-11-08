#include "AvgPoolTestLayer.hpp"

#include "AvgPoolTestBuffer.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

AvgPoolTestLayer::AvgPoolTestLayer(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *AvgPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, AvgPoolTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
