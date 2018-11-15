#include "GateAvgPoolTestLayer.hpp"

#include "GateAvgPoolTestBuffer.hpp"
#include <components/GSynAccumulator.hpp>
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

GateAvgPoolTestLayer::GateAvgPoolTestLayer(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *GateAvgPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     GateAvgPoolTestBuffer>(getName(), parameters(), mCommunicator);
}

} /* namespace PV */
