#include "GateSumPoolTestLayer.hpp"

#include "GateSumPoolTestBuffer.hpp"
#include <components/GSynAccumulator.hpp>
#include <components/HyPerActivityComponent.hpp>
#include <components/HyPerInternalStateBuffer.hpp>

namespace PV {

GateSumPoolTestLayer::GateSumPoolTestLayer(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *GateSumPoolTestLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     GateSumPoolTestBuffer>(getName(), parameters(), mCommunicator);
}

} /* namespace PV */
