#include "PvpListLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/PvpListActivityBuffer.hpp"

namespace PV {

PvpListLayer::PvpListLayer(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

PvpListLayer::~PvpListLayer() {}

void PvpListLayer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   InputLayer::initialize(name, params, comm);
}

ActivityComponent *PvpListLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<PvpListActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
