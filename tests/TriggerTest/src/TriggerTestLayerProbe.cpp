/*
 * TriggerTestLayer.cpp
 * Author: slundquist
 */

#include "TriggerTestLayerProbe.hpp"
#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/Messages.hpp>
#include <components/InputActivityBuffer.hpp>
#include <components/PhaseParam.hpp>
#include <io/PVParams.hpp>
#include <observerpattern/BaseMessage.hpp>
#include <observerpattern/Response.hpp>
#include <probes/ProbeTriggerComponent.hpp>
#include <probes/TargetLayerComponent.hpp>
#include <utils/PVLog.hpp>

#include <cmath>
#include <functional>
#include <string>

namespace PV {
TriggerTestLayerProbe::TriggerTestLayerProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

Response::Status TriggerTestLayerProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   status = status + mProbeTargetLayerLocator->communicateInitInfo(message);
   status = status + mProbeTrigger->communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   char const *inputLayerName = mProbeTrigger->getTriggerLayerName();
   if (inputLayerName) {
      auto *inputBuffer = message->mObjectTable->findObject<InputActivityBuffer>(inputLayerName);
      FatalIf(
            inputBuffer == nullptr,
            "%s: triggerLayerName \"%s\" is not an InputLayer-derived object, as required "
            "by TriggerTest.\n",
            getDescription_c(),
            inputLayerName);
      FatalIf(
            inputBuffer->getDisplayPeriod() != mInputDisplayPeriod,
            "%s: triggerLayer \"%s\" has display period %d, "
            "but TriggerTest requires displayPeriod = %d.\n",
            getDescription_c(),
            inputLayerName,
            inputBuffer->getDisplayPeriod(),
            mInputDisplayPeriod);
   }
   return Response::SUCCESS;
}

void TriggerTestLayerProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeTargetLayerLocator = std::make_shared<TargetLayerComponent>(name, params);
   mProbeTrigger            = std::make_shared<ProbeTriggerComponent>(name, params);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   BaseObject::initialize(name, params, comm);
}

void TriggerTestLayerProbe::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerOutputStateMessage const>(msgptr);
      return respondLayerOutputState(castMessage);
   };
   mMessageActionMap.emplace("LayerOutputState", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ProbeWriteParamsMessage const>(msgptr);
      return respondProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ProbeWriteParams", action);
}

int TriggerTestLayerProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseObject::ioParamsFillGroup(ioFlag);
   mProbeTargetLayerLocator->ioParamsFillGroup(ioFlag);
   mProbeTrigger->ioParamsFillGroup(ioFlag);
   return status;
}

Response::Status
TriggerTestLayerProbe::outputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   double simTime   = message->mTime;
   double deltaTime = message->mDeltaTime;
   if (simTime < deltaTime / 2.0) {
      return Response::SUCCESS;
   }

   std::string name(getName());
   char const *name_c = name.c_str();
   bool updateNeeded  = mProbeTrigger->needUpdate(simTime, deltaTime);
   int stepNumber     = static_cast<int>(std::nearbyint(simTime / deltaTime));

   InfoLog().printf(
         "%s: time=%f, dt=%f, needUpdate=%d, triggerOffset=%f\n",
         name.c_str(),
         simTime,
         deltaTime,
         updateNeeded,
         mProbeTrigger->getTriggerOffset());
   // 4 different layers
   if (name == "notriggerlayerprobe") {
      // No trigger, always update
      FatalIf(updateNeeded != true, "Test failed at %s. Expected true, found false.\n", name_c);
   }
   else if (name == "trigger0layerprobe") {
      // Trigger with offset of 0, assuming display period is 5
      if ((stepNumber - 1) % mInputDisplayPeriod == 0) {
         FatalIf(updateNeeded != true, "Test failed at %s. Expected true, found false.\n", name_c);
      }
      else {
         FatalIf(updateNeeded != false, "Test failed at %s. Expected false, found true.\n", name_c);
      }
   }
   else if (name == "trigger1layerprobe") {
      // Trigger with offset of 1, assuming display period is 5
      if (stepNumber % mInputDisplayPeriod == 0) {
         FatalIf(updateNeeded != true, "Test failed at %s. Expected true, found false.\n", name_c);
      }
      else {
         FatalIf(updateNeeded != false, "Test failed at %s. Expected false, found true.\n", name_c);
      }
   }
   // Trigger with offset of 1, assuming display period is 5
   else if (name == "trigger2layerprobe") {
      if ((stepNumber + 1) % mInputDisplayPeriod == 0) {
         FatalIf(updateNeeded != true, "Test failed at %s. Expected true, found false.\n", name_c);
      }
      else {
         FatalIf(updateNeeded != false, "Test failed at %s. Expected false, found true.\n", name_c);
      }
   }
   return Response::SUCCESS;
}

Response::Status TriggerTestLayerProbe::respondLayerOutputState(
      std::shared_ptr<LayerOutputStateMessage const> message) {
   auto status          = Response::SUCCESS;
   auto *targetLayer    = mProbeTargetLayerLocator->getTargetLayer();
   int targetLayerPhase = targetLayer->getComponentByType<PhaseParam>()->getPhase();
   if (message->mPhase == targetLayerPhase) {
      status = outputState(message);
   }
   return status;
}

Response::Status TriggerTestLayerProbe::respondProbeWriteParams(
      std::shared_ptr<ProbeWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

} // namespace PV
