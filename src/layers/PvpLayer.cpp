#include "PvpLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/PvpActivityBuffer.hpp"

namespace PV {

PvpLayer::PvpLayer(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

PvpLayer::~PvpLayer() {}

void PvpLayer::initialize(char const *name, PVParams *params, Communicator *comm) {
   InputLayer::initialize(name, params, comm);
}

ActivityComponent *PvpLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<PvpActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
