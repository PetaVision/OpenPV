/*
 * LIF.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 */

#include "LIF.hpp"
#include "components/LIFActivityComponent.hpp"

namespace PV {

LIF::LIF(const char *name, PVParams *params, Communicator *comm) { initialize(name, params, comm); }

LIF::LIF() {}

LIF::~LIF() {}

void LIF::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *LIF::createActivityComponent() {
   return new LIFActivityComponent(getName(), parameters(), mCommunicator);
}

} // end namespace PV
