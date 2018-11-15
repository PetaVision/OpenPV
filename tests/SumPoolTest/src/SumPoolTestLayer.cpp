#include "SumPoolTestLayer.hpp"

#include "SumPoolTestBuffer.hpp"
#include <components/GSynAccumulator.hpp>
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

SumPoolTestLayer::SumPoolTestLayer(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *SumPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, HyPerInternalStateBuffer, SumPoolTestBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
