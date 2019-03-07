/*
 * LIFGap.cpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#include "LIFGap.hpp"
#include "components/LIFGapActivityComponent.hpp"

namespace PV {

LIFGap::LIFGap(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

LIFGap::LIFGap() {}

LIFGap::~LIFGap() {}

void LIFGap::initialize(const char *name, PVParams *params, Communicator const *comm) {
   LIF::initialize(name, params, comm);
}

ActivityComponent *LIFGap::createActivityComponent() {
   return new LIFGapActivityComponent(getName(), parameters(), mCommunicator);
}

} // end namespace PV
