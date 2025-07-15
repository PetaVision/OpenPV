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

IndexLayer::IndexLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   initialize(paramsIO, comm);
}

IndexLayer::~IndexLayer() {}

void IndexLayer::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

ActivityComponent *IndexLayer::createActivityComponent() {
   // IndexInternalState isn't a CloneV-type InternalState, but it doesn't use GSyn,
   // so the CloneActivityComponent class template does what we need.
   return new CloneActivityComponent<IndexInternalState, HyPerActivityBuffer>(
         mParamsIO, mCommunicator);
}

} // end namespace PV
