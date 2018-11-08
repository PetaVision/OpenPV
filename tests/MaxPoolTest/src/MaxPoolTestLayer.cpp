#include "MaxPoolTestLayer.hpp"

#include "MaxPoolTestBuffer.hpp"
#include <columns/HyPerCol.hpp>
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

MaxPoolTestLayer::MaxPoolTestLayer(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *MaxPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, MaxPoolTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
