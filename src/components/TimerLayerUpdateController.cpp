#include "TimerLayerUpdateController.hpp"

namespace PV {

TimerLayerUpdateController::TimerLayerUpdateController(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

TimerLayerUpdateController::TimerLayerUpdateController() {}

TimerLayerUpdateController::~TimerLayerUpdateController() {}

void TimerLayerUpdateController::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   LayerUpdateController::initialize(name, params, comm);
}

void TimerLayerUpdateController::setObjectType() { mObjectType = "TimerLayerUpdateController"; }

int TimerLayerUpdateController::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_timerPeriod(ioFlag);
   return PV_SUCCESS;
}

void TimerLayerUpdateController::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      mTriggerLayerName = nullptr;
      mTriggerFlag      = false;
      parameters()->handleUnnecessaryStringParameter(
            getName(), "triggerLayerName", nullptr /*correct value*/);
   }
}

void TimerLayerUpdateController::ioParam_timerPeriod(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "timerPeriod", &mDeltaUpdateTime);
   FatalIf(
         ioFlag == PARAMS_IO_READ and mDeltaUpdateTime <= 0.0,
         "Layer \"%s\" has timerPeriod = %f; this value must be positive.\n",
         getName(), mDeltaUpdateTime);
}

void TimerLayerUpdateController::initMessageActionMap() {
   BaseObject::initMessageActionMap();
}

Response::Status TimerLayerUpdateController::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   return status;
}

Response::Status
TimerLayerUpdateController::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   mLastUpdateTime = message->mDeltaTime;
   return Response::SUCCESS;
}

bool TimerLayerUpdateController::needUpdate(double simTime, double deltaTime) const {
   double deltaUpdateTime = mDeltaUpdateTime;
   double numUpdates      = (simTime - mLastUpdateTime) / deltaUpdateTime;
   double closest         = std::fabs(numUpdates - std::nearbyint(numUpdates)) * deltaUpdateTime;
   bool updateNeeded      = closest < 0.5 * deltaTime;
   return updateNeeded;
}

} // namespace PV
