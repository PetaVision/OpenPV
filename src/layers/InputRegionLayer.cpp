/*
 * InputRegionLayer.cpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#include "InputRegionLayer.hpp"
#include "components/BasePublisherComponent.hpp"
#include "components/DependentBoundaryConditions.hpp"
#include "components/DependentPhaseParam.hpp"
#include "components/InputRegionActivityComponent.hpp"

namespace PV {

InputRegionLayer::InputRegionLayer(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   initialize(paramsIO, comm);
}

InputRegionLayer::InputRegionLayer() {}

InputRegionLayer::~InputRegionLayer() {}

void InputRegionLayer::initialize(std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) {
   HyPerLayer::initialize(paramsIO, comm);
}

void InputRegionLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

PhaseParam *InputRegionLayer::createPhaseParam() {
   return new DependentPhaseParam(mParamsIO, mCommunicator);
}

BoundaryConditions *InputRegionLayer::createBoundaryConditions() {
   return new DependentBoundaryConditions(mParamsIO, mCommunicator);
}

LayerUpdateController *InputRegionLayer::createLayerUpdateController() { return nullptr; }

LayerInputBuffer *InputRegionLayer::createLayerInput() { return nullptr; }

ActivityComponent *InputRegionLayer::createActivityComponent() {
   return new InputRegionActivityComponent(mParamsIO, mCommunicator);
}

BasePublisherComponent *InputRegionLayer::createPublisher() {
   return new BasePublisherComponent(mParamsIO, mCommunicator);
}

OriginalLayerNameParam *InputRegionLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO, mCommunicator);
}

} /* namespace PV */
