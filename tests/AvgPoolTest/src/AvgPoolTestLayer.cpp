#include "AvgPoolTestLayer.hpp"

#include "AvgPoolTestBuffer.hpp"
#include <components/GSynAccumulator.hpp>
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

AvgPoolTestLayer::AvgPoolTestLayer(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *AvgPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, HyPerInternalStateBuffer, AvgPoolTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
