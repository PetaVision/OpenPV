#include "GateMaxPoolTestLayer.hpp"

#include "GateMaxPoolTestBuffer.hpp"
#include <components/ANNActivityBuffer.hpp>
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

GateMaxPoolTestLayer::GateMaxPoolTestLayer(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *GateMaxPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GateMaxPoolTestBuffer,
                                     HyPerInternalStateBuffer,
                                     ANNActivityBuffer>(getName(), parameters(), mCommunicator);
}

} /* namespace PV */
