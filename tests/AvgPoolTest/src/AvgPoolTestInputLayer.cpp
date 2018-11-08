#include "AvgPoolTestInputLayer.hpp"

#include "AvgPoolTestInputBuffer.hpp"
#include <components/ActivityComponentActivityOnly.hpp>

namespace PV {

AvgPoolTestInputLayer::AvgPoolTestInputLayer(
      const char *name,
      PVParams *params,
      Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *AvgPoolTestInputLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<AvgPoolTestInputBuffer>(
         getName(), parameters(), mCommunicator);
}

} /* namespace PV */
