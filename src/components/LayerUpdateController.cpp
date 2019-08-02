/*
 * LayerUpdateController.cpp
 *
 *  Created on: Nov 20, 2018
 *      Author: peteschultz
 */

#include "LayerUpdateController.hpp"
#include "components/InternalStateBuffer.hpp"
#include <cmath>

namespace PV {

LayerUpdateController::LayerUpdateController(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

LayerUpdateController::LayerUpdateController() {}

LayerUpdateController::~LayerUpdateController() {
   free(mTriggerLayerName);
   free(mTriggerBehavior);
   free(mTriggerResetLayerName);
}

void LayerUpdateController::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void LayerUpdateController::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerClearProgressFlagsMessage const>(msgptr);
      return respondLayerClearProgressFlags(castMessage);
   };
   mMessageActionMap.emplace("LayerClearProgressFlags", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerRecvSynapticInputMessage const>(msgptr);
      return respondLayerRecvSynapticInput(castMessage);
   };
   mMessageActionMap.emplace("LayerRecvSynapticInput", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerUpdateStateMessage const>(msgptr);
      return respondLayerUpdateState(castMessage);
   };
   mMessageActionMap.emplace("LayerUpdateState", action);
}

void LayerUpdateController::setObjectType() { mObjectType = "LayerUpdateController"; }

int LayerUpdateController::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_triggerFlag(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_triggerBehavior(ioFlag);
   ioParam_triggerResetLayerName(ioFlag);
   return PV_SUCCESS;
}

// triggerFlag was deprecated Aug 7, 2015 and marked obsolete Nov 20, 2018.
// Setting triggerLayerName to a nonempty string has the effect of triggerFlag=true, and
// setting triggerLayerName to NULL or "" has the effect of triggerFlag=false.
void LayerUpdateController::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ && parameters()->present(name, "triggerFlag")) {
      FatalIf(
            parameters()->present(name, "triggerFlag"),
            "%s sets triggerFlag, but this flag is obsolete.\n"
            "   If triggerLayerName is a nonempty string, triggering will be on;\n"
            "   if triggerLayerName is empty or null, triggering will be off.\n",
            getDescription_c());
   }
}

void LayerUpdateController::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, name, "triggerLayerName", &mTriggerLayerName, NULL, false /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      FatalIf(
            mTriggerLayerName and !strcmp(name, mTriggerLayerName),
            "%s triggerLayerName cannot be the same as the name of the layer itself.\n",
            getDescription_c());
   }
   mTriggerFlag = mTriggerLayerName != nullptr and mTriggerLayerName[0] != '\0';
}

void LayerUpdateController::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (mTriggerFlag) {
      parameters()->ioParamValue(ioFlag, name, "triggerOffset", &mTriggerOffset, mTriggerOffset);
   }
}

void LayerUpdateController::ioParam_triggerBehavior(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (mTriggerFlag) {
      parameters()->ioParamString(
            ioFlag,
            name,
            "triggerBehavior",
            &mTriggerBehavior,
            "updateOnlyOnTrigger",
            true /*warnIfAbsent*/);
      if (mTriggerBehavior == NULL or mTriggerBehavior[0] == '\0') {
         free(mTriggerBehavior);
         mTriggerBehavior = strdup("updateOnlyOnTrigger");
      }
      if (!strcmp(mTriggerBehavior, "updateOnlyOnTrigger")) {
         mTriggerBehaviorType = UPDATEONLYONTRIGGER;
      }
      else if (!strcmp(mTriggerBehavior, "resetStateOnTrigger")) {
         mTriggerBehaviorType = RESETSTATEONTRIGGER;
      }
      else if (!strcmp(mTriggerBehavior, "ignore")) {
         mTriggerBehaviorType = NO_TRIGGER;
      }
      else {
         Fatal().printf(
               "%s triggerBehavior=\"%s\" is unrecognized.\n",
               getDescription_c(),
               mTriggerBehavior);
      }
   }
}

void LayerUpdateController::ioParam_triggerResetLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (mTriggerFlag) {
      pvAssert(!parameters()->presentAndNotBeenRead(name, "triggerBehavior"));
      if (mTriggerBehaviorType == RESETSTATEONTRIGGER) {
         parameters()->ioParamStringRequired(
               ioFlag, name, "triggerResetLayerName", &mTriggerResetLayerName);
      }
   }
}

Response::Status LayerUpdateController::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *objectTable = message->mObjectTable;

   mPhaseParam = objectTable->findObject<PhaseParam>(getName());
   FatalIf(mPhaseParam == nullptr, "%s requires a PhaseParam component.\n", getDescription_c());

   mLayerInput = objectTable->findObject<LayerInputBuffer>(getName());
   // It is not an error for mLayerInput to be null.

   mActivityComponent = objectTable->findObject<ActivityComponent>(getName());
   FatalIf(
         mActivityComponent == nullptr, "%s requires an ActivityComponent.\n", getDescription_c());

   if (mTriggerFlag) {
      setTriggerUpdateController(objectTable);
      if (mTriggerBehaviorType == RESETSTATEONTRIGGER) {
         setTriggerResetComponent(objectTable);
         auto *componentV = mActivityComponent->getComponentByType<InternalStateBuffer>();
         FatalIf(
               componentV == nullptr,
               "%s uses resetStateOnTrigger but does not have an InternalState.\n",
               getDescription_c());
         // applyTrigger will check that componentV can be written to. We don't do it here because
         // ReadWritePointer isn't set until the component's CommunicateInitInfo stage, and
         // we can avoid a postpone.
      }
   }
   return Response::SUCCESS;
}

void LayerUpdateController::setTriggerUpdateController(ObserverTable const *table) {
   if (mTriggerUpdateController != nullptr) {
      return;
   }
   mTriggerUpdateController = table->findObject<LayerUpdateController>(mTriggerLayerName);
   FatalIf(
         mTriggerUpdateController == nullptr,
         "%s triggerLayerName \"%s\" does not have a LayerUpdateController component.\n",
         getDescription_c(),
         mTriggerLayerName);
}

void LayerUpdateController::setTriggerResetComponent(ObserverTable const *table) {
   char const *resetLayerName = nullptr; // Will point to name of actual resetLayer, whether
   // triggerResetLayerName is blank (in which case resetLayerName==triggerLayerName) or not
   if (mTriggerResetLayerName == nullptr or mTriggerResetLayerName[0] == '\0') {
      resetLayerName = mTriggerLayerName;
   }
   else {
      resetLayerName = mTriggerResetLayerName;
   }

   mTriggerResetComponent = table->findObject<ActivityComponent>(resetLayerName);
   FatalIf(
         mTriggerResetComponent == nullptr,
         "%s triggerResetLayerName points to \"%s\", which has no ActivityComponent.\n",
         getDescription_c(),
         resetLayerName);
}

Response::Status LayerUpdateController::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   message->mDataRegistry->registerCheckpointData(
         std::string(getName()),
         std::string("lastUpdateTime"),
         &mLastUpdateTime,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   return Response::SUCCESS;
}

Response::Status
LayerUpdateController::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   if (mTriggerBehaviorType == UPDATEONLYONTRIGGER) {
      pvAssert(mTriggerFlag and mTriggerUpdateController);
      if (!mTriggerUpdateController->getInitialValuesSetFlag()) {
         return Response::POSTPONE;
      }
      mDeltaUpdateTime = mTriggerUpdateController->getDeltaUpdateTime();
   }
   else {
      setNontriggerDeltaUpdateTime(message->mDeltaTime);
   }
   mLastUpdateTime = message->mDeltaTime;
   return Response::SUCCESS;
}

void LayerUpdateController::setNontriggerDeltaUpdateTime(double deltaTime) {
   mDeltaUpdateTime = deltaTime;
}

Response::Status LayerUpdateController::respondLayerClearProgressFlags(
      std::shared_ptr<LayerClearProgressFlagsMessage const> message) {
   mHasReceived = false;
   mHasUpdated  = false;
   return Response::SUCCESS;
}

Response::Status LayerUpdateController::respondLayerRecvSynapticInput(
      std::shared_ptr<LayerRecvSynapticInputMessage const> message) {
   if (!mLayerInput) {
      return Response::NO_ACTION;
   }
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != mPhaseParam->getPhase()) {
      return status;
   }
   if (mHasReceived) {
      return status;
   }
   if (!needUpdate(message->mTime, message->mDeltaT)) {
      return status;
   }
   if (*(message->mSomeLayerHasActed)) {
      *(message->mSomeLayerIsPending) = true;
      return status;
   }

   message->mTimer->start();
   status = mLayerInput->respond(message);
   if (status == Response::SUCCESS) {
      mHasReceived                   = true;
      *(message->mSomeLayerHasActed) = true;
   }
   message->mTimer->stop();
   return status;
}

Response::Status LayerUpdateController::respondLayerUpdateState(
      std::shared_ptr<LayerUpdateStateMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != mPhaseParam->getPhase()) {
      return status;
   }
   if (mHasUpdated) {
      return status;
   }
#ifdef PV_USE_CUDA
   if (mLayerInput and message->mRecvOnGpuFlag != mLayerInput->isUsingGPU()) {
      return status;
   }
   if (!mLayerInput and message->mRecvOnGpuFlag != mActivityComponent->getUpdateGpu()) {
      return status;
   }
   if (message->mUpdateOnGpuFlag != mActivityComponent->getUpdateGpu()) {
      return status;
   }
#endif // PV_USE_CUDA
   if (needUpdate(message->mTime, message->mDeltaT)) {
      if (*(message->mSomeLayerHasActed)) {
         *(message->mSomeLayerIsPending) = true;
         return status;
      }
      if (mLayerInput and !mHasReceived) {
         *(message->mSomeLayerIsPending) = true;
         return status;
      }
      applyTrigger(message->mTime, message->mDeltaT);
      mActivityComponent->updateState(message->mTime, message->mDeltaT);

      mHasUpdated                    = true;
      mLastUpdateTime                = message->mTime;
      *(message->mSomeLayerHasActed) = true;
   }
   return status;
}

bool LayerUpdateController::needUpdate(double simTime, double deltaTime) const {
   bool updateNeeded = false;
   if (mTriggerUpdateController != nullptr and mTriggerBehaviorType == UPDATEONLYONTRIGGER) {
      updateNeeded = mTriggerUpdateController->needUpdate(simTime + mTriggerOffset, deltaTime);
   }
   else {
      double deltaUpdateTime = getDeltaUpdateTime();
      if (deltaUpdateTime <= 0) {
         updateNeeded = false;
      }
      else {
         double numUpdates = (simTime - mLastUpdateTime) / deltaUpdateTime;
         double closest    = std::fabs(numUpdates - std::nearbyint(numUpdates)) * deltaUpdateTime;
         updateNeeded      = closest < 0.5 * deltaTime;
      }
   }
   return updateNeeded;
}

void LayerUpdateController::applyTrigger(double simTime, double deltaTime) {
   if (!mTriggerFlag or mTriggerBehaviorType != RESETSTATEONTRIGGER) {
      return;
   }
   bool resetNeeded = mTriggerUpdateController->needUpdate(simTime + mTriggerOffset, deltaTime);
   if (!resetNeeded) {
      return;
   }

   // If TriggerResetLayer has a V component, copy it to this layer's V component.
   // If not, copy the TriggerResetLayer's A component to this layer's V component.
   pvAssert(mTriggerResetComponent != nullptr); // Was set in CommunicateInitInfo.
   auto *componentV = mActivityComponent->getComponentByType<InternalStateBuffer>();
   pvAssert(componentV); // Was checked in CommunicateInitInfo
   float *V = componentV->getReadWritePointer();
   FatalIf(
         V == nullptr,
         "%s uses resetStateOnTrigger but the InternalState is not writeable.\n",
         getDescription_c());

   auto *resetComponentV = mTriggerResetComponent->getComponentByType<InternalStateBuffer>();
   if (resetComponentV) {
      float const *resetV = resetComponentV->getBufferData();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
      for (int k = 0; k < componentV->getBufferSizeAcrossBatch(); k++) {
         V[k] = resetV[k];
      }
   }
   else {
      auto *resetComponentA = mTriggerResetComponent->getComponentByType<ActivityBuffer>();
      pvAssert(resetComponentA);
      float const *resetA   = resetComponentA->getBufferData();
      PVLayerLoc const *loc = resetComponentA->getLayerLoc();
      PVHalo const *halo    = &loc->halo;
      for (int b = 0; b < loc->nbatch; b++) {
         float const *resetABatch = resetA + (b * resetComponentA->getBufferSize());
         int const numNeurons     = componentV->getBufferSize();
         float *VBatch            = V + (b * numNeurons);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < numNeurons; k++) {
            int kex = kIndexExtended(
                  k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
            VBatch[k] = resetABatch[kex];
         }
      }
   }
}

} // namespace PV
