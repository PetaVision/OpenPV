/*
 * CloneVLayer.cpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#include "CloneVLayer.hpp"
#include "components/CloneActivityComponent.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/CloneLayerGeometry.hpp"
#include "components/HyPerActivityBuffer.hpp"

namespace PV {

CloneVLayer::CloneVLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

CloneVLayer::CloneVLayer() {
   // initialize() gets called by subclass's initialize method
}

CloneVLayer::~CloneVLayer() {}

void CloneVLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

void CloneVLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *CloneVLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

LayerGeometry *CloneVLayer::createLayerGeometry() {
   return new CloneLayerGeometry(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

LayerInputBuffer *CloneVLayer::createLayerInput() { return nullptr; }

ActivityComponent *CloneVLayer::createActivityComponent() {
   return new CloneActivityComponent<CloneInternalStateBuffer, HyPerActivityBuffer>(
         mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} /* namespace PV */
