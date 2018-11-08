#include "SumPoolTestInputLayer.hpp"

#include "SumPoolTestInputBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

SumPoolTestInputLayer::SumPoolTestInputLayer(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *SumPoolTestInputLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<SumPoolTestInputBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
