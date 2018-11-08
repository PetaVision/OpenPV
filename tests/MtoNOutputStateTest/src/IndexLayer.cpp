/**
 * IndexLayer.cpp
 *
 *  Created on: Mar 3, 2017
 *      Author: peteschultz
 *
 */

#include "IndexLayer.hpp"

#include "IndexInternalState.hpp"
#include <components/ActivityComponentWithInternalState.hpp>
#include <components/HyPerActivityBuffer.hpp>

namespace PV {

IndexLayer::IndexLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

IndexLayer::~IndexLayer() {}

void IndexLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

ActivityComponent *IndexLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<IndexInternalState, HyPerActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
