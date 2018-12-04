/*
 * InputRegionLayer.cpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#include "InputRegionLayer.hpp"
#include "components/DenseLayerOutputComponent.hpp"
#include "components/DependentBoundaryConditions.hpp"
#include "components/DependentPhaseParam.hpp"
#include "components/InputRegionActivityComponent.hpp"

namespace PV {

InputRegionLayer::InputRegionLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

InputRegionLayer::InputRegionLayer() {}

InputRegionLayer::~InputRegionLayer() {}

void InputRegionLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   HyPerLayer::initialize(name, params, comm);
}

void InputRegionLayer::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

PhaseParam *InputRegionLayer::createPhaseParam() {
   return new DependentPhaseParam(name, parameters(), mCommunicator);
}

BoundaryConditions *InputRegionLayer::createBoundaryConditions() {
   return new DependentBoundaryConditions(name, parameters(), mCommunicator);
}

LayerUpdateController *InputRegionLayer::createLayerUpdateController() { return nullptr; }

LayerInputBuffer *InputRegionLayer::createLayerInput() { return nullptr; }

ActivityComponent *InputRegionLayer::createActivityComponent() {
   return new InputRegionActivityComponent(getName(), parameters(), mCommunicator);
}

LayerOutputComponent *InputRegionLayer::createLayerOutput() {
   return new DenseLayerOutputComponent(getName(), parameters(), mCommunicator);
}

OriginalLayerNameParam *InputRegionLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parameters(), mCommunicator);
}

} /* namespace PV */
