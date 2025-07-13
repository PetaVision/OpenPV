/*
 * InputLayerUpdateController.cpp
 *
 *  Created on: Nov 20, 2018
 *      Author: peteschultz
 */

#include "InputLayerUpdateController.hpp"
#include "components/InputActivityBuffer.hpp"

namespace PV {

InputLayerUpdateController::InputLayerUpdateController(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InputLayerUpdateController::InputLayerUpdateController() {}

InputLayerUpdateController::~InputLayerUpdateController() {}

void InputLayerUpdateController::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   LayerUpdateController::initialize(params, defaults, comm);
}

void InputLayerUpdateController::setObjectType() { mObjectType = "InputLayerUpdateController"; }

void InputLayerUpdateController::ioParam_triggerLayerName(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      mTriggerLayerName = "";
      mTriggerFlag      = false;
      mParamsIO->handleUnnecessaryParameter("triggerLayerName", std::string("") /*correct value*/);
   }
}

void InputLayerUpdateController::setNontriggerDeltaUpdateTime(double deltaTime) {
   auto *activityBuffer = mActivityComponent->getComponentByType<InputActivityBuffer>();
   pvAssert(activityBuffer);
   auto displayPeriod = activityBuffer->getDisplayPeriod();
   mDeltaUpdateTime   = displayPeriod > 0 ? displayPeriod : DBL_MAX;
}

} // namespace PV
