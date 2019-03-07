/*
 * GapLayer.cpp
 *
 *  Created on: May 11, 2011
 *      Author: garkenyon
 */

#include "GapLayer.hpp"
#include "components/CloneActivityComponent.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/GapActivityBuffer.hpp"

namespace PV {

GapLayer::GapLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

GapLayer::~GapLayer() {}

void GapLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   CloneVLayer::initialize(name, params, comm);
}

ActivityComponent *GapLayer::createActivityComponent() {
   return new CloneActivityComponent<CloneInternalStateBuffer, GapActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
