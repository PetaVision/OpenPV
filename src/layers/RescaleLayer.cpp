/*
 * RescaleLayer.cpp
 */

#include "RescaleLayer.hpp"
#include "components/ActivityComponentActivityOnly.hpp"
#include "components/RescaleActivityBuffer.hpp"

namespace PV {

RescaleLayer::RescaleLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

RescaleLayer::RescaleLayer() {}

RescaleLayer::~RescaleLayer() {}

void RescaleLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   CloneVLayer::initialize(name, params, comm);
}

ActivityComponent *RescaleLayer::createActivityComponent() {
   return new ActivityComponentActivityOnly<RescaleActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
