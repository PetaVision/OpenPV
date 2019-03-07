#include "MaxPoolTestLayer.hpp"

#include "GateMaxPoolTestBuffer.hpp"
#include "MaxPoolTestBuffer.hpp"
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

MaxPoolTestLayer::MaxPoolTestLayer(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *MaxPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, HyPerInternalStateBuffer, MaxPoolTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
