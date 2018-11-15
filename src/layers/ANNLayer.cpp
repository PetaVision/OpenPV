/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"
#include "components/ANNActivityBuffer.hpp"
#include "components/GSynAccumulator.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"

namespace PV {

ANNLayer::ANNLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

ANNLayer::~ANNLayer() {}

void ANNLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *ANNLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator, HyPerInternalStateBuffer, ANNActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
