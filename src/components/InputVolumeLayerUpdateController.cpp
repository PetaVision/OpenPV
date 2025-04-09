/*
 * InputVolumeLayerUpdateController.cpp
 */

#include "InputVolumeLayerUpdateController.hpp"
#include "components/InputVolumeActivityBuffer.hpp"

namespace PV {

InputVolumeLayerUpdateController::InputVolumeLayerUpdateController(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InputVolumeLayerUpdateController::InputVolumeLayerUpdateController() {}

InputVolumeLayerUpdateController::~InputVolumeLayerUpdateController() {}

void InputVolumeLayerUpdateController::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   LayerUpdateController::initialize(name, params, comm);
}

void InputVolumeLayerUpdateController::setObjectType() {
   mObjectType = "InputVolumeLayerUpdateController";
}

void InputVolumeLayerUpdateController::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mTriggerLayerName = nullptr;
      mTriggerFlag      = false;
      parameters()->handleUnnecessaryStringParameter(
            getName(), "triggerLayerName", nullptr /*correct value*/);
   }
}

void InputVolumeLayerUpdateController::setNontriggerDeltaUpdateTime(double deltaTime) {
   auto *activityBuffer = mActivityComponent->getComponentByType<InputVolumeActivityBuffer>();
   pvAssert(activityBuffer);
   auto displayPeriod = activityBuffer->getDisplayPeriod();
   mDeltaUpdateTime   = displayPeriod > 0 ? displayPeriod : DBL_MAX;
}

} // namespace PV
