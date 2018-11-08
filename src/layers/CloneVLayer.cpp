/*
 * CloneVLayer.cpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#include "CloneVLayer.hpp"
#include "components/ActivityComponentWithInternalState.hpp"
#include "components/CloneInternalStateBuffer.hpp"
#include "components/HyPerActivityBuffer.hpp"

namespace PV {

CloneVLayer::CloneVLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

CloneVLayer::CloneVLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

CloneVLayer::~CloneVLayer() {}

int CloneVLayer::initialize_base() { return PV_SUCCESS; }

void CloneVLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

void CloneVLayer::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

OriginalLayerNameParam *CloneVLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parameters(), mCommunicator);
}

LayerInputBuffer *CloneVLayer::createLayerInput() { return nullptr; }

ActivityComponent *CloneVLayer::createActivityComponent() {
   return new ActivityComponentWithInternalState<CloneInternalStateBuffer, HyPerActivityBuffer>(
         name, parameters(), mCommunicator);
}

} /* namespace PV */
