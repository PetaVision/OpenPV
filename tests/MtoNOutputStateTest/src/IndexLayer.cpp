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

IndexLayer::IndexLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

IndexLayer::~IndexLayer() {}

void IndexLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

ActivityComponent *IndexLayer::createActivityComponent() {
   // IndexInternalState isn't a CloneV-type InternalState, but it doesn't use GSyn,
   // so the CloneActivityComponent class template does what we need.
   return new CloneActivityComponent<IndexInternalState, HyPerActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} // end namespace PV
