/*
 * DropoutLayer.cpp
 */

#include "DropoutLayer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/DropoutActivityBuffer.hpp"
#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

DropoutLayer::DropoutLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

DropoutLayer::~DropoutLayer() {}

void DropoutLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *DropoutLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, DropoutActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
