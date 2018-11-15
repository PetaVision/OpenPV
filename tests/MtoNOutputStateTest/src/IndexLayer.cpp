/**
 * IndexLayer.cpp
 *
 *  Created on: Mar 3, 2017
 *      Author: peteschultz
 *
 */

#include "IndexLayer.hpp"

#include "IndexInternalState.hpp"
#include <components/CloneActivityComponent.hpp>
#include <components/GSynAccumulator.hpp>
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
   // IndexInternalState isn't a CloneV-type InternalState, but it doesn't use GSyn,
   // so the CloneActivityComponent class template does what we need.
   return new CloneActivityComponent<IndexInternalState, HyPerActivityBuffer>(
         getName(), parameters(), mCommunicator);
}

} // end namespace PV
