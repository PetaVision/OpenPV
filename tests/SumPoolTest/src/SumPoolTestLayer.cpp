#include "SumPoolTestLayer.hpp"

#include "SumPoolTestBuffer.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

SumPoolTestLayer::SumPoolTestLayer(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *SumPoolTestLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, SumPoolTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
