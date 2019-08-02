/*
 * CloneVLayer.cpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#include "CloneVLayer.hpp"
#include "components/CloneActivityComponent.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/HyPerActivityBuffer.hpp"

namespace PV {

CloneVLayer::CloneVLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

CloneVLayer::CloneVLayer() {
   // initialize() gets called by subclass's initialize method
}

CloneVLayer::~CloneVLayer() {}

void CloneVLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

void CloneVLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

OriginalLayerNameParam *CloneVLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parameters(), mCommunicator);
}

LayerInputBuffer *CloneVLayer::createLayerInput() { return nullptr; }

ActivityComponent *CloneVLayer::createActivityComponent() {
   return new CloneActivityComponent<CloneInternalStateBuffer, HyPerActivityBuffer>(
         name, parameters(), mCommunicator);
}

} /* namespace PV */
