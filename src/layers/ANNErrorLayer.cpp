/*
 * ANNErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNErrorLayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/ErrScaleInternalStateBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"

namespace PV {

ANNErrorLayer::ANNErrorLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ANNErrorLayer::~ANNErrorLayer() {}

void ANNErrorLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *ANNErrorLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     ErrScaleInternalStateBuffer,
                                     ANNActivityBuffer>(getName(), parameters(), mCommunicator);
}

} // end namespace PV
