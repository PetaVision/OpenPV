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
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InputLayerUpdateController::InputLayerUpdateController() {}

InputLayerUpdateController::~InputLayerUpdateController() {}

void InputLayerUpdateController::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   LayerUpdateController::initialize(name, params, comm);
}

void InputLayerUpdateController::setObjectType() { mObjectType = "InputLayerUpdateController"; }

void InputLayerUpdateController::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mTriggerLayerName = nullptr;
      mTriggerFlag      = false;
      parameters()->handleUnnecessaryStringParameter(
            getName(), "triggerLayerName", nullptr /*correct value*/);
   }
}

void InputLayerUpdateController::setNontriggerDeltaUpdateTime(double deltaTime) {
   auto *activityBuffer = mActivityComponent->getComponentByType<InputActivityBuffer>();
   pvAssert(activityBuffer);
   auto displayPeriod = activityBuffer->getDisplayPeriod();
   mDeltaUpdateTime   = displayPeriod > 0 ? displayPeriod : DBL_MAX;
}

} // namespace PV
