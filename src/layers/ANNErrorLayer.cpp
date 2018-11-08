/*
 * ANNErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNErrorLayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/ErrScaleInternalStateBuffer.hpp"

namespace PV {

ANNErrorLayer::ANNErrorLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

ANNErrorLayer::~ANNErrorLayer() {}

void ANNErrorLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *ANNErrorLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<ErrScaleInternalStateBuffer, ANNActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
