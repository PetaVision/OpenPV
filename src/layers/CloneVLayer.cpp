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

CloneVLayer::CloneVLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

CloneVLayer::CloneVLayer() {
   // initialize() gets called by subclass's initialize method
}

CloneVLayer::~CloneVLayer() {}

void CloneVLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

void CloneVLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *CloneVLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO, mCommunicator);
}

LayerGeometry *CloneVLayer::createLayerGeometry() {
   return new CloneLayerGeometry(mParamsIO, mCommunicator);
}

LayerInputBuffer *CloneVLayer::createLayerInput() { return nullptr; }

ActivityComponent *CloneVLayer::createActivityComponent() {
   return new CloneActivityComponent<CloneInternalStateBuffer, HyPerActivityBuffer>(
         mParamsIO, mCommunicator);
}

} /* namespace PV */
