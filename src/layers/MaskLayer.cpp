
/*
 * MaskLayer.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist
 */

#include "MaskLayer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/MaskActivityBuffer.hpp"

namespace PV {

MaskLayer::MaskLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

MaskLayer::~MaskLayer() {}

void MaskLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *MaskLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<HyPerInternalStateBuffer, MaskActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
