/*
 * GapLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "GapLayer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/GapActivityBuffer.hpp"

namespace PV {

GapLayer::GapLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

GapLayer::~GapLayer() {}

void GapLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   CloneVLayer::initialize(name, params, comm);
}

ActivityComponent *GapLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<CloneInternalStateBuffer, GapActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
