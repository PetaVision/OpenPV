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

InputRegionLayer::InputRegionLayer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InputRegionLayer::InputRegionLayer() {}

InputRegionLayer::~InputRegionLayer() {}

void InputRegionLayer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   HyPerLayer::initialize(params, defaults, comm);
}

void InputRegionLayer::fillComponentTable() {
   HyPerLayer::fillComponentTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam);
   }
}

PhaseParam *InputRegionLayer::createPhaseParam() {
   return new DependentPhaseParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

BoundaryConditions *InputRegionLayer::createBoundaryConditions() {
   return new DependentBoundaryConditions(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

LayerUpdateController *InputRegionLayer::createLayerUpdateController() { return nullptr; }

LayerInputBuffer *InputRegionLayer::createLayerInput() { return nullptr; }

ActivityComponent *InputRegionLayer::createActivityComponent() {
   return new InputRegionActivityComponent(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

BasePublisherComponent *InputRegionLayer::createPublisher() {
   return new BasePublisherComponent(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

OriginalLayerNameParam *InputRegionLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(mParamsIO->getParams(), mParamsIO->getDefaults(), mCommunicator);
}

} /* namespace PV */
