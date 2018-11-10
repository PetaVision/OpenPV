/*
 * InputRegionLayer.cpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#include "InputRegionLayer.hpp"
#include "components/DependentBoundaryConditions.hpp"
#include "components/DependentPhaseParam.hpp"
#include "components/InputRegionActivityComponent.hpp"

namespace PV {

InputRegionLayer::InputRegionLayer(const char *name, HyPerCol *hc) { initialize(name, hc); }

InputRegionLayer::InputRegionLayer() {}

InputRegionLayer::~InputRegionLayer() {}

int InputRegionLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void InputRegionLayer::createComponentTable(char const *description) {
   HyPerLayer::createComponentTable(description);
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

PhaseParam *InputRegionLayer::createPhaseParam() { return new DependentPhaseParam(name, parent); }

BoundaryConditions *InputRegionLayer::createBoundaryConditions() {
   return new DependentBoundaryConditions(name, parent);
}

LayerInputBuffer *InputRegionLayer::createLayerInput() { return nullptr; }

ActivityComponent *InputRegionLayer::createActivityComponent() {
   return new InputRegionActivityComponent(getName(), parent);
}

OriginalLayerNameParam *InputRegionLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

void InputRegionLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      triggerLayerName = nullptr;
      triggerFlag      = false;
      parameters()->handleUnnecessaryParameter(name, "triggerLayerName");
   }
}

void InputRegionLayer::ioParam_sparseLayer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      sparseLayer = false;
      parameters()->handleUnnecessaryParameter(name, "sparseLayer");
   }
}

bool InputRegionLayer::needUpdate(double timed, double dt) const { return false; }

} /* namespace PV */
