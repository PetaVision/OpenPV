/*
 * HyPerLayer.cpp
 *
 *  Created on: Jul 29, 2008
 *
 *  The top of the hierarchy for layer classes.
 *
 *  To make it easy to subclass from classes in the HyPerLayer hierarchy,
 *  please follow the guidelines below when adding subclasses to the HyPerLayer hierarchy:
 *
 *  For a class named DerivedLayer that is derived from a class named BaseLayer,
 *  the .hpp file should have
*/

#include "HyPerLayer.hpp"
#include "checkpointing/CheckpointEntryPvpBuffer.hpp"
#include "checkpointing/CheckpointEntryRandState.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "connections/BaseConnection.hpp"
#include "include/default_params.h"
#include "include/pv_common.h"
#include "io/FileStream.hpp"
#include "io/io.hpp"
#include <assert.h>
#include <iostream>
#include <sstream>
#include <string.h>

namespace PV {

// This constructor is protected so that only derived classes can call it.
// It should be called as the normal method of object construction by
// derived classes.  It should NOT call any virtual methods
HyPerLayer::HyPerLayer() { initialize_base(); }

HyPerLayer::HyPerLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

// initialize_base should be called only by constructors.  It should not
// call any virtual methods, because polymorphism is not available when
// a base class constructor is inherited from a derived class constructor.
// In general, initialize_base should be used only to initialize member variables
// to safe values.
int HyPerLayer::initialize_base() {
   name                  = NULL;
   probes                = NULL;
   numProbes             = 0;
   marginIndices         = NULL;
   numMargin             = 0;
   writeTime             = 0;
   initialWriteTime      = 0;
   triggerFlag           = false; // Default to update every timestamp
   triggerLayer          = NULL;
   triggerLayerName      = NULL;
   triggerBehavior       = NULL;
   triggerBehaviorType   = NO_TRIGGER;
   triggerResetLayerName = NULL;
   triggerOffset         = 0;

#ifdef PV_USE_CUDA
   allocDeviceDatastore     = false;
   allocDeviceActiveIndices = false;
   d_Datastore              = NULL;
   d_ActiveIndices          = NULL;
   d_numActive              = NULL;
   updatedDeviceActivity    = true; // Start off always updating activity
   updatedDeviceDatastore   = true;
   updatedDeviceGSyn        = true;
   mUpdateGpu               = false;
   krUpdate                 = NULL;
#ifdef PV_USE_CUDNN
   cudnn_GSyn      = NULL;
   cudnn_Datastore = NULL;
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

   update_timer    = NULL;
   publish_timer   = NULL;
   timescale_timer = NULL;
   io_timer        = NULL;

#ifdef PV_USE_CUDA
   gpu_update_timer = NULL;
#endif

   recvConns.clear();

   return PV_SUCCESS;
}

///////
/// Classes derived from HyPerLayer should call HyPerLayer::initialize themselves
/// to take advantage of virtual methods.  Note that the HyPerLayer constructor
/// does not call initialize.  This way, HyPerLayer::initialize can call virtual
/// methods and the derived class's method will be the one that gets called.
int HyPerLayer::initialize(const char *name, HyPerCol *hc) {
   int status = ComponentBasedObject::initialize(name, hc);
   if (status != PV_SUCCESS) {
      return status;
   }

   writeTime                = initialWriteTime;
   writeActivityCalls       = 0;
   writeActivitySparseCalls = 0;
   numDelayLevels = 1; // If a connection has positive delay so that more delay levels are needed,
   // numDelayLevels is increased when BaseConnection::communicateInitInfo calls
   // increaseDelayLevels

   mLastUpdateTime  = 0.0;
   mLastTriggerTime = 0.0;
   return PV_SUCCESS;
}

void HyPerLayer::initMessageActionMap() {
   ComponentBasedObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerSetMaxPhaseMessage const>(msgptr);
      return respondLayerSetMaxPhase(castMessage);
   };
   mMessageActionMap.emplace("LayerSetMaxPhase", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerWriteParamsMessage const>(msgptr);
      return respondLayerWriteParams(castMessage);
   };
   mMessageActionMap.emplace("LayerWriteParams", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerProbeWriteParamsMessage const>(msgptr);
      return respondLayerProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("LayerProbeWriteParams", action);

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

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerCopyFromGpuMessage const>(msgptr);
      return respondLayerCopyFromGpu(castMessage);
   };
   mMessageActionMap.emplace("LayerCopyFromGpu", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerAdvanceDataStoreMessage const>(msgptr);
      return respondLayerAdvanceDataStore(castMessage);
   };
   mMessageActionMap.emplace("LayerAdvanceDataStore", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerPublishMessage const>(msgptr);
      return respondLayerPublish(castMessage);
   };
   mMessageActionMap.emplace("LayerPublish", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerCheckNotANumberMessage const>(msgptr);
      return respondLayerCheckNotANumber(castMessage);
   };
   mMessageActionMap.emplace("LayerCheckNotANumber", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerOutputStateMessage const>(msgptr);
      return respondLayerOutputState(castMessage);
   };
   mMessageActionMap.emplace("LayerOutputState", action);
}

void HyPerLayer::setObserverTable() {
   mLayerGeometry = createLayerGeometry();
   if (mLayerGeometry) {
      addUniqueComponent(mLayerGeometry->getDescription(), mLayerGeometry);
   }
   mPhaseParam = createPhaseParam();
   if (mPhaseParam) {
      addUniqueComponent(mPhaseParam->getDescription(), mPhaseParam);
   }
   mBoundaryConditions = createBoundaryConditions();
   if (mBoundaryConditions) {
      addUniqueComponent(mBoundaryConditions->getDescription(), mBoundaryConditions);
   }
   auto *initializeFromCheckpointComponent = createInitializeFromCheckpointFlag();
   if (initializeFromCheckpointComponent) {
      addUniqueComponent(
            initializeFromCheckpointComponent->getDescription(), initializeFromCheckpointComponent);
   }
   mLayerInput = createLayerInput();
   if (mLayerInput) {
      addUniqueComponent(mLayerInput->getDescription(), mLayerInput);
   }
   mInternalState = createInternalState();
   if (mInternalState) {
      addUniqueComponent(mInternalState->getDescription(), mInternalState);
   }
   mActivity = createActivity();
   if (mActivity) {
      addUniqueComponent(mActivity->getDescription(), mActivity);
   }
}

LayerGeometry *HyPerLayer::createLayerGeometry() { return new LayerGeometry(name, parent); }

PhaseParam *HyPerLayer::createPhaseParam() { return new PhaseParam(name, parent); }

BoundaryConditions *HyPerLayer::createBoundaryConditions() {
   return new BoundaryConditions(name, parent);
}

InitializeFromCheckpointFlag *HyPerLayer::createInitializeFromCheckpointFlag() {
   return new InitializeFromCheckpointFlag(name, parent);
}

LayerInputBuffer *HyPerLayer::createLayerInput() { return new LayerInputBuffer(name, parent); }

InternalStateBuffer *HyPerLayer::createInternalState() {
   return new InternalStateBuffer(name, parent);
}

ActivityBuffer *HyPerLayer::createActivity() { return new ActivityBuffer(name, parent); }

HyPerLayer::~HyPerLayer() {
   delete update_timer;
   delete publish_timer;
   delete timescale_timer;
   delete io_timer;
#ifdef PV_USE_CUDA
   delete gpu_update_timer;
#endif

   delete mOutputStateStream;

#ifdef PV_USE_CUDA
   if (krUpdate) {
      delete krUpdate;
   }
   if (d_Datastore) {
      delete d_Datastore;
   }

#ifdef PV_USE_CUDNN
   if (cudnn_Datastore) {
      delete cudnn_Datastore;
   }
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

   free(marginIndices);
   free(probes); // All probes are deleted by the HyPerCol, so probes[i] doesn't need to be deleted,
   // only the array itself.

   free(triggerLayerName);
   free(triggerBehavior);
   free(triggerResetLayerName);

   delete publisher;
}

template <typename T>
int HyPerLayer::freeBuffer(T **buf) {
   free(*buf);
   *buf = NULL;
   return PV_SUCCESS;
}
// Declare the instantiations of allocateBuffer that occur in other .cpp files; otherwise you may
// get linker errors.
template int HyPerLayer::freeBuffer<float>(float **buf);
template int HyPerLayer::freeBuffer<int>(int **buf);

int HyPerLayer::freeRestrictedBuffer(float **buf) { return freeBuffer(buf); }

int HyPerLayer::freeExtendedBuffer(float **buf) { return freeBuffer(buf); }

template <typename T>
void HyPerLayer::allocateBuffer(T **buf, int bufsize, const char *bufname) {
   *buf = (T *)calloc(bufsize, sizeof(T));
   if (*buf == NULL) {
      Fatal().printf(
            "%s: rank %d process unable to allocate memory for %s: %s.\n",
            getDescription_c(),
            parent->getCommunicator()->globalCommRank(),
            bufname,
            strerror(errno));
   }
}
// Declare the instantiations of allocateBuffer that occur in other .cpp files; otherwise you may
// get linker errors.
template void HyPerLayer::allocateBuffer<float>(float **buf, int bufsize, const char *bufname);
template void HyPerLayer::allocateBuffer<int>(int **buf, int bufsize, const char *bufname);

void HyPerLayer::allocateRestrictedBuffer(float **buf, char const *bufname) {
   allocateBuffer(buf, getNumNeuronsAllBatches(), bufname);
}

void HyPerLayer::allocateExtendedBuffer(float **buf, char const *bufname) {
   allocateBuffer(buf, getNumExtendedAllBatches(), bufname);
}

void HyPerLayer::allocateBuffers() {
   // Kept so that LIF, etc. can add additional buffers. Will go away as these buffers
   // are converted to BufferComponent objects.
}

void HyPerLayer::addPublisher() {
   MPIBlock const *mpiBlock  = parent->getCommunicator()->getLocalMPIBlock();
   PVLayerCube *activityCube = (PVLayerCube *)malloc(sizeof(PVLayerCube));
   activityCube->numItems    = getNumExtendedAllBatches();
   activityCube->data        = mActivity->getActivity();
   activityCube->loc         = *getLayerLoc();
   publisher = new Publisher(*mpiBlock, activityCube, getNumDelayLevels(), getSparseFlag());
}

void HyPerLayer::checkpointPvpActivityFloat(
      Checkpointer *checkpointer,
      char const *bufferName,
      float *pvpBuffer,
      bool extended) {
   bool registerSucceeded = checkpointer->registerCheckpointEntry(
         std::make_shared<CheckpointEntryPvpBuffer<float>>(
               getName(),
               bufferName,
               checkpointer->getMPIBlock(),
               pvpBuffer,
               getLayerLoc(),
               extended),
         false /*not constant*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         getDescription_c(),
         bufferName);
}

void HyPerLayer::checkpointRandState(
      Checkpointer *checkpointer,
      char const *bufferName,
      Random *randState,
      bool extendedFlag) {
   bool registerSucceeded = checkpointer->registerCheckpointEntry(
         std::make_shared<CheckpointEntryRandState>(
               getName(),
               bufferName,
               checkpointer->getMPIBlock(),
               randState->getRNG(0),
               getLayerLoc(),
               extendedFlag),
         false /*not constant*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         getDescription_c(),
         bufferName);
}

Response::Status
HyPerLayer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   if (mInternalState) {
      mInternalState->respond(message);
   }
   if (triggerLayer and triggerBehaviorType == UPDATEONLY_TRIGGER) {
      if (!triggerLayer->getInitialValuesSetFlag()) {
         return Response::POSTPONE;
      }
      mDeltaUpdateTime = triggerLayer->getDeltaUpdateTime();
   }
   else {
      setNontriggerDeltaUpdateTime(message->mDeltaTime);
   }
   initializeActivity();
   mLastUpdateTime  = message->mDeltaTime;
   mLastTriggerTime = message->mDeltaTime;
   return Response::SUCCESS;
}

void HyPerLayer::setNontriggerDeltaUpdateTime(double dt) { mDeltaUpdateTime = dt; }

#ifdef PV_USE_CUDA
Response::Status HyPerLayer::copyInitialStateToGPU() {
   if (mUpdateGpu) {
      float *h_V = getV();
      if (mInternalState != nullptr) {
         pvAssert(mInternalState->isUsingGPU());
         mInternalState->copyToCuda();
      }
      pvAssert(mActivity->isUsingGPU());
      mActivity->copyToCuda();
   }
   return Response::SUCCESS;
}
#endif // PV_USE_CUDA

void HyPerLayer::initializeActivity() {
   int status = setActivity();
   FatalIf(status != PV_SUCCESS, "%s failed to initialize activity.\n", getDescription_c());
}

int HyPerLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   // Derived classes with new params behavior should override ioParamsFillGroup
   // and the overriding method should call the base class's ioParamsFillGroup.
   for (auto &c : mObserverTable) {
      auto obj = dynamic_cast<BaseObject *>(c);
      obj->ioParams(ioFlag, false, false);
   }
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerFlag(ioFlag);
   ioParam_triggerOffset(ioFlag);
   ioParam_triggerBehavior(ioFlag);
   ioParam_triggerResetLayerName(ioFlag);
   ioParam_writeStep(ioFlag);
   ioParam_initialWriteTime(ioFlag);
   ioParam_sparseLayer(ioFlag);

   // GPU-specific parameter.  If not using GPUs, this flag
   // can be set to false or left out, but it is an error
   // to set updateGpu to true if compiling without GPUs.
   ioParam_updateGpu(ioFlag);

   ioParam_dataType(ioFlag);
   return PV_SUCCESS;
}

// The dataType parameter was marked obsolete Mar 29, 2018.
// Only TransposePoolingConn made use of the dataType, by checking
// that the post index layer is PV_INT. But mPostIndexLayer has type
// PoolingIndexLayer*, which is automatically PV_INT. So the dataType
// check has become vacuous.
void HyPerLayer::ioParam_dataType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ and parameters()->stringPresent(getName(), "dataType")) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         WarnLog().printf(
               "%s defines the dataType param, which is no longer used.\n", getDescription_c());
      }
   }
}

void HyPerLayer::ioParam_updateGpu(enum ParamsIOFlag ioFlag) {
#ifdef PV_USE_CUDA
   parameters()->ioParamValue(
         ioFlag, name, "updateGpu", &mUpdateGpu, mUpdateGpu, true /*warnIfAbsent*/);
   mUsingGPUFlag = mUpdateGpu;
#else // PV_USE_CUDA
   bool mUpdateGpu = false;
   parameters()->ioParamValue(
         ioFlag, name, "updateGpu", &mUpdateGpu, mUpdateGpu, false /*warnIfAbsent*/);
   if (parent->getCommunicator()->globalCommRank() == 0) {
      FatalIf(
            mUpdateGpu,
            "%s: updateGpu is set to true, but PetaVision was compiled without GPU acceleration.\n",
            getDescription_c());
   }
#endif // PV_USE_CUDA
}

void HyPerLayer::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag, name, "triggerLayerName", &triggerLayerName, NULL, false /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      if (triggerLayerName && !strcmp(name, triggerLayerName)) {
         if (parent->getCommunicator()->commRank() == 0) {
            ErrorLog().printf(
                  "%s: triggerLayerName cannot be the same as the name of the layer itself.\n",
                  getDescription_c());
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      triggerFlag = (triggerLayerName != NULL && triggerLayerName[0] != '\0');
   }
}

// triggerFlag was deprecated Aug 7, 2015.
// Setting triggerLayerName to a nonempty string has the effect of triggerFlag=true, and
// setting triggerLayerName to NULL or "" has the effect of triggerFlag=false.
// While triggerFlag is being deprecated, it is an error for triggerFlag to be false
// and triggerLayerName to be a nonempty string.
void HyPerLayer::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (ioFlag == PARAMS_IO_READ && parameters()->present(name, "triggerFlag")) {
      bool flagFromParams = false;
      parameters()->ioParamValue(ioFlag, name, "triggerFlag", &flagFromParams, flagFromParams);
      if (parent->getCommunicator()->globalCommRank() == 0) {
         WarnLog(triggerFlagMessage);
         triggerFlagMessage.printf("%s: triggerFlag has been deprecated.\n", getDescription_c());
         triggerFlagMessage.printf(
               "   If triggerLayerName is a nonempty string, triggering will be on;\n");
         triggerFlagMessage.printf(
               "   if triggerLayerName is empty or null, triggering will be off.\n");
         if (flagFromParams != triggerFlag) {
            ErrorLog(errorMessage);
            errorMessage.printf("triggerLayerName=", name);
            if (triggerLayerName) {
               errorMessage.printf("\"%s\"", triggerLayerName);
            }
            else {
               errorMessage.printf("NULL");
            }
            errorMessage.printf(
                  " implies triggerFlag=%s but triggerFlag was set in params to %s\n",
                  triggerFlag ? "true" : "false",
                  flagFromParams ? "true" : "false");
         }
      }
      if (flagFromParams != triggerFlag) {
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
}

void HyPerLayer::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      parameters()->ioParamValue(ioFlag, name, "triggerOffset", &triggerOffset, triggerOffset);
      if (triggerOffset < 0) {
         if (parent->getCommunicator()->globalCommRank() == 0) {
            Fatal().printf(
                  "%s: TriggerOffset (%f) must be positive\n", getDescription_c(), triggerOffset);
         }
      }
   }
}
void HyPerLayer::ioParam_triggerBehavior(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      parameters()->ioParamString(
            ioFlag,
            name,
            "triggerBehavior",
            &triggerBehavior,
            "updateOnlyOnTrigger",
            true /*warnIfAbsent*/);
      if (triggerBehavior == NULL || !strcmp(triggerBehavior, "")) {
         free(triggerBehavior);
         triggerBehavior     = strdup("updateOnlyOnTrigger");
         triggerBehaviorType = UPDATEONLY_TRIGGER;
      }
      else if (!strcmp(triggerBehavior, "updateOnlyOnTrigger")) {
         triggerBehaviorType = UPDATEONLY_TRIGGER;
      }
      else if (!strcmp(triggerBehavior, "resetStateOnTrigger")) {
         triggerBehaviorType = RESETSTATE_TRIGGER;
      }
      else if (!strcmp(triggerBehavior, "ignore")) {
         triggerBehaviorType = NO_TRIGGER;
      }
      else {
         if (parent->getCommunicator()->commRank() == 0) {
            ErrorLog().printf(
                  "%s: triggerBehavior=\"%s\" is unrecognized.\n",
                  getDescription_c(),
                  triggerBehavior);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
   }
   else {
      triggerBehaviorType = NO_TRIGGER;
   }
}

void HyPerLayer::ioParam_triggerResetLayerName(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "triggerLayerName"));
   if (triggerFlag) {
      assert(!parameters()->presentAndNotBeenRead(name, "triggerBehavior"));
      if (!strcmp(triggerBehavior, "resetStateOnTrigger")) {
         parameters()->ioParamStringRequired(
               ioFlag, name, "triggerResetLayerName", &triggerResetLayerName);
      }
   }
}

void HyPerLayer::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "writeStep", &writeStep, parent->getDeltaTime());
}

void HyPerLayer::ioParam_initialWriteTime(enum ParamsIOFlag ioFlag) {
   assert(!parameters()->presentAndNotBeenRead(name, "writeStep"));
   if (writeStep >= 0.0) {
      parameters()->ioParamValue(ioFlag, name, "initialWriteTime", &initialWriteTime, 0.0);
      if (ioFlag == PARAMS_IO_READ && writeStep > 0.0 && initialWriteTime < 0.0) {
         double storeInitialWriteTime = initialWriteTime;
         while (initialWriteTime < 0.0) {
            initialWriteTime += writeStep;
         }
         if (parent->getCommunicator()->globalCommRank() == 0) {
            WarnLog(warningMessage);
            warningMessage.printf(
                  "%s: initialWriteTime %f is negative.  Adjusting "
                  "initialWriteTime:\n",
                  getDescription_c(),
                  initialWriteTime);
            warningMessage.printf("    initialWriteTime adjusted to %f\n", initialWriteTime);
         }
      }
   }
}

void HyPerLayer::ioParam_sparseLayer(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "sparseLayer", &sparseLayer, false);
}

Response::Status
HyPerLayer::respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message) {
   return notify(message, parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
}

Response::Status
HyPerLayer::respondLayerWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

Response::Status HyPerLayer::respondLayerProbeWriteParams(
      std::shared_ptr<LayerProbeWriteParamsMessage const> message) {
   return outputProbeParams();
}

Response::Status HyPerLayer::respondLayerClearProgressFlags(
      std::shared_ptr<LayerClearProgressFlagsMessage const> message) {
   if (mLayerInput) {
      mLayerInput->respond(message);
   }
   mHasUpdated = false;
   return Response::SUCCESS;
}

Response::Status HyPerLayer::respondLayerRecvSynapticInput(
      std::shared_ptr<LayerRecvSynapticInputMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
   message->mTimer->start();
   if (needUpdate(message->mTime, message->mDeltaT)) {
      auto *layerInputBuffer = getComponentByType<LayerInputBuffer>();
      if (layerInputBuffer) {
         layerInputBuffer->respond(message);
      }
   }
   message->mTimer->stop();

   return status;
}

Response::Status
HyPerLayer::respondLayerUpdateState(std::shared_ptr<LayerUpdateStateMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
#ifdef PV_USE_CUDA
   if (mLayerInput and message->mRecvOnGpuFlag != mLayerInput->isUsingGPU()) {
      return status;
   }
   if (!mLayerInput and message->mRecvOnGpuFlag != mUpdateGpu) {
      return status;
   }
   if (message->mUpdateOnGpuFlag != mUpdateGpu) {
      return status;
   }
#endif // PV_USE_CUDA
   if (mHasUpdated or !needUpdate(message->mTime, message->mDeltaT)) {
      return status;
   }
   if (*(message->mSomeLayerHasActed)) {
      *(message->mSomeLayerIsPending) = true;
      return status;
   }
   if (mLayerInput and !mLayerInput->getHasReceived()) {
      *(message->mSomeLayerIsPending) = true;
      return status;
   }
   // If we're here, layer has not done UpdateState this timestep, but is ready to do so.
   status = callUpdateState(message->mTime, message->mDeltaT);

   mHasUpdated                    = true;
   *(message->mSomeLayerHasActed) = true;
   return status;
}

#ifdef PV_USE_CUDA
Response::Status
HyPerLayer::respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
   message->mTimer->start();
   if (mActivity and mActivity->isUsingGPU()) {
      mActivity->copyFromCuda();
   }
   if (mInternalState and mInternalState->isUsingGPU()) {
      mInternalState->copyFromCuda();
   }
   if (mLayerInput and mLayerInput->isUsingGPU()) {
      mLayerInput->respond(message);
   }
   if (mUpdateGpu) {
      gpu_update_timer->accumulateTime();
   }
   message->mTimer->stop();
   return status;
}
#endif // PV_USE_CUDA

Response::Status HyPerLayer::respondLayerAdvanceDataStore(
      std::shared_ptr<LayerAdvanceDataStoreMessage const> message) {
   if (message->mPhase < 0 || message->mPhase == getPhase()) {
      publisher->increaseTimeLevel();
   }
   return Response::SUCCESS;
}

Response::Status
HyPerLayer::respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message) {
   if (message->mPhase != getPhase()) {
      return Response::NO_ACTION;
   }
   publish(parent->getCommunicator(), message->mTime);
   return Response::SUCCESS;
}

Response::Status HyPerLayer::respondLayerCheckNotANumber(
      std::shared_ptr<LayerCheckNotANumberMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
   auto layerData = getLayerData();
   int const N    = getNumExtendedAllBatches();
   for (int n = 0; n < N; n++) {
      float a = layerData[n];
      FatalIf(
            a != a,
            "%s has not-a-number values in the activity buffer.  Exiting.\n",
            getDescription_c());
   }
   return status;
}

Response::Status
HyPerLayer::respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
   status = outputState(message->mTime, message->mDeltaTime); // also calls probes' outputState
   return status;
}

Response::Status HyPerLayer::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   Response::Status status = ComponentBasedObject::setCudaDevice(message);
   return notify(message, parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
}

#ifdef PV_USE_CUDA

int HyPerLayer::allocateUpdateKernel() {
   Fatal() << "Layer \"" << name << "\" of type " << mObjectType
           << " does not support updating on gpus yet\n";
   return PV_FAILURE;
}

/**
 * Allocate GPU buffers.
 */
int HyPerLayer::allocateDeviceBuffers() {
   int status = 0;

   const size_t size    = getNumNeuronsAllBatches() * sizeof(float);
   const size_t size_ex = getNumExtendedAllBatches() * sizeof(float);

   PVCuda::CudaDevice *device = mCudaDevice;

   if (allocDeviceDatastore) {
      d_Datastore = device->createBuffer(size_ex, &getDescription());
      assert(d_Datastore);
#ifdef PV_USE_CUDNN
      cudnn_Datastore = device->createBuffer(size_ex, &getDescription());
      assert(cudnn_Datastore);
#endif
   }

   if (allocDeviceActiveIndices) {
      int const nbatch = mLayerGeometry->getLayerLoc()->nbatch;
      d_numActive      = device->createBuffer(nbatch * sizeof(long), &getDescription());
      d_ActiveIndices  = device->createBuffer(
            getNumExtendedAllBatches() * sizeof(SparseList<float>::Entry), &getDescription());
      assert(d_ActiveIndices);
   }

#ifdef PV_USE_CUDNN
   // mLayerInput's CudaBuffer is the entire GSyn buffer. cudnn_GSyn is only one gsyn channel
   if (mLayerInput and mLayerInput->isUsingGPU()) {
      cudnn_GSyn = device->createBuffer(size, &getDescription());
   }
#endif

   return status;
}

#endif // PV_USE_CUDA

Response::Status
HyPerLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectMapComponent = getComponentByType<ObjectMapComponent>();
   if (!objectMapComponent) {
      objectMapComponent = new ObjectMapComponent(name, parent);
      objectMapComponent->setObjectMap(message->mHierarchy);
      addUniqueComponent(objectMapComponent->getDescription(), objectMapComponent);
      // ObserverTable takes ownership; objectMapComponent will be deleted by
      // Subject::deleteObserverTable() method during destructor.
   }
   pvAssert(objectMapComponent);

   auto communicateMessage = std::make_shared<CommunicateInitInfoMessage>(
         mObserverTable.getObjectMap(),
         message->mNxGlobal,
         message->mNyGlobal,
         message->mNBatchGlobal,
         message->mNumThreads);

   Response::Status status =
         notify(communicateMessage, parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
   if (!Response::completed(status)) {
      return status;
   }

   PVLayerLoc const *loc = getLayerLoc();

   auto *initializeFromCheckpointComponent = getComponentByType<InitializeFromCheckpointFlag>();
   mInitializeFromCheckpointFlag =
         initializeFromCheckpointComponent->getInitializeFromCheckpointFlag();

   if (triggerFlag) {
      triggerLayer = message->lookup<HyPerLayer>(std::string(triggerLayerName));
      if (triggerLayer == NULL) {
         if (parent->getCommunicator()->commRank() == 0) {
            ErrorLog().printf(
                  "%s: triggerLayerName \"%s\" is not a layer in the HyPerCol.\n",
                  getDescription_c(),
                  triggerLayerName);
         }
         MPI_Barrier(parent->getCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }
      if (triggerBehaviorType == RESETSTATE_TRIGGER) {
         char const *resetLayerName = NULL; // Will point to name of actual resetLayer, whether
         // triggerResetLayerName is blank (in which case
         // resetLayerName==triggerLayerName) or not
         if (triggerResetLayerName == NULL || triggerResetLayerName[0] == '\0') {
            resetLayerName    = triggerLayerName;
            triggerResetLayer = triggerLayer;
         }
         else {
            resetLayerName    = triggerResetLayerName;
            triggerResetLayer = message->lookup<HyPerLayer>(std::string(triggerResetLayerName));
            if (triggerResetLayer == NULL) {
               if (parent->getCommunicator()->commRank() == 0) {
                  ErrorLog().printf(
                        "%s: triggerResetLayerName \"%s\" is not a layer in the HyPerCol.\n",
                        getDescription_c(),
                        triggerResetLayerName);
               }
               MPI_Barrier(parent->getCommunicator()->communicator());
               exit(EXIT_FAILURE);
            }
         }
         if (!triggerResetLayer->getInitInfoCommunicatedFlag()) {
            return Response::POSTPONE;
         }
         // Check that triggerResetLayer and this layer have the same (restricted) dimensions.
         // Do we need to postpone until triggerResetLayer has finished its communicateInitInfo?
         PVLayerLoc const *triggerLoc = triggerResetLayer->getLayerLoc();
         PVLayerLoc const *localLoc   = this->getLayerLoc();
         if (triggerLoc->nxGlobal != localLoc->nxGlobal
             || triggerLoc->nyGlobal != localLoc->nyGlobal
             || triggerLoc->nf != localLoc->nf) {
            if (parent->getCommunicator()->commRank() == 0) {
               Fatal(errorMessage);
               errorMessage.printf(
                     "%s: triggerResetLayer \"%s\" has incompatible dimensions.\n",
                     getDescription_c(),
                     resetLayerName);
               errorMessage.printf(
                     "    \"%s\" is %d-by-%d-by-%d and \"%s\" is %d-by-%d-by-%d.\n",
                     name,
                     localLoc->nxGlobal,
                     localLoc->nyGlobal,
                     localLoc->nf,
                     resetLayerName,
                     triggerLoc->nxGlobal,
                     triggerLoc->nyGlobal,
                     triggerLoc->nf);
            }
         }
      }
   }

#ifdef PV_USE_CUDA
   // Here, the connection tells all participating recv layers to allocate memory on gpu
   // if receive from gpu is set. These buffers should be set in allocate
   if (mUpdateGpu) {
      if (mLayerInput) {
         mLayerInput->useCuda();
      }
      if (mInternalState) {
         mInternalState->useCuda();
      }
      if (mActivity) {
         mActivity->useCuda();
      }
   }
#endif

   return Response::SUCCESS;
}

int HyPerLayer::openOutputStateFile(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   pvAssert(writeStep >= 0);

   auto *checkpointer = message->mDataRegistry;
   if (checkpointer->getMPIBlock()->getRank() == 0) {
      std::string outputStatePath(getName());
      outputStatePath.append(".pvp");

      std::string checkpointLabel(getName());
      checkpointLabel.append("_filepos");

      bool createFlag    = checkpointer->getCheckpointReadDirectory().empty();
      mOutputStateStream = new CheckpointableFileStream(
            outputStatePath.c_str(), createFlag, checkpointer, checkpointLabel);
      mOutputStateStream->respond(message); // CheckpointableFileStream needs to register data
   }
   return PV_SUCCESS;
}

void HyPerLayer::synchronizeMarginWidth(HyPerLayer *layer) {
   if (layer == this) {
      return;
   }
   auto *thisLayerGeometry  = getComponentByType<LayerGeometry>();
   auto *otherLayerGeometry = layer->getComponentByType<LayerGeometry>();
   pvAssert(thisLayerGeometry != nullptr);
   pvAssert(otherLayerGeometry != nullptr);
   thisLayerGeometry->synchronizeMarginWidth(otherLayerGeometry);

   return;
}

int HyPerLayer::equalizeMargins(HyPerLayer *layer1, HyPerLayer *layer2) {
   int border1, border2, maxborder, result;
   int status = PV_SUCCESS;

   border1   = layer1->getLayerLoc()->halo.lt;
   border2   = layer2->getLayerLoc()->halo.lt;
   maxborder = border1 > border2 ? border1 : border2;
   layer1->requireMarginWidth(maxborder, &result, 'x');
   if (result != maxborder) {
      status = PV_FAILURE;
   }
   layer2->requireMarginWidth(maxborder, &result, 'x');
   if (result != maxborder) {
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      Fatal().printf(
            "Error in rank %d process: unable to synchronize x-margin widths of layers \"%s\" and "
            "\"%s\" to %d\n",
            layer1->parent->getCommunicator()->globalCommRank(),
            layer1->getName(),
            layer2->getName(),
            maxborder);
      ;
   }
   assert(
         layer1->getLayerLoc()->halo.lt == layer2->getLayerLoc()->halo.lt
         && layer1->getLayerLoc()->halo.rt == layer2->getLayerLoc()->halo.rt
         && layer1->getLayerLoc()->halo.lt == layer1->getLayerLoc()->halo.rt
         && layer1->getLayerLoc()->halo.lt == maxborder);

   border1   = layer1->getLayerLoc()->halo.dn;
   border2   = layer2->getLayerLoc()->halo.dn;
   maxborder = border1 > border2 ? border1 : border2;
   layer1->requireMarginWidth(maxborder, &result, 'y');
   if (result != maxborder) {
      status = PV_FAILURE;
   }
   layer2->requireMarginWidth(maxborder, &result, 'y');
   if (result != maxborder) {
      status = PV_FAILURE;
   }
   if (status != PV_SUCCESS) {
      Fatal().printf(
            "Error in rank %d process: unable to synchronize y-margin widths of layers \"%s\" and "
            "\"%s\" to %d\n",
            layer1->parent->getCommunicator()->globalCommRank(),
            layer1->getName(),
            layer2->getName(),
            maxborder);
      ;
   }
   assert(
         layer1->getLayerLoc()->halo.dn == layer2->getLayerLoc()->halo.dn
         && layer1->getLayerLoc()->halo.up == layer2->getLayerLoc()->halo.up
         && layer1->getLayerLoc()->halo.dn == layer1->getLayerLoc()->halo.up
         && layer1->getLayerLoc()->halo.dn == maxborder);
   return status;
}

Response::Status HyPerLayer::allocateDataStructures() {
   // Once initialize and communicateInitInfo have been called, HyPerLayer has the
   // information it needs to allocate the membrane potential buffer V, the
   // activity buffer activity->data, and the data store.
   auto status = Response::SUCCESS;

   auto allocateMessage = std::make_shared<AllocateDataStructuresMessage>();
   notify(allocateMessage, parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);

   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   PVHalo const *halo    = &loc->halo;

   // If not mirroring, fill the boundaries with the value in the valueBC param
   if (!mBoundaryConditions->getMirrorBCflag() && mBoundaryConditions->getValueBC() != 0.0f) {
      mBoundaryConditions->applyBoundaryConditions(mActivity->getActivity(), getLayerLoc());
   }

   // allocate storage for the input conductance arrays
   allocateBuffers();

// Allocate cuda stuff on gpu if set
#ifdef PV_USE_CUDA
   int deviceStatus = allocateDeviceBuffers();
   // Allocate receive from post kernel
   if (deviceStatus == 0) {
      status = Response::SUCCESS;
   }
   else {
      Fatal().printf(
            "%s unable to allocate device memory in rank %d process: %s\n",
            getDescription_c(),
            parent->getCommunicator()->globalCommRank(),
            strerror(errno));
   }
   if (mUpdateGpu) {
      // This function needs to be overwritten as needed on a subclass basis
      deviceStatus = allocateUpdateKernel();
      if (deviceStatus == 0) {
         status = Response::SUCCESS;
      }
   }
#endif

   addPublisher();

   return status;
}

/*
 * Call this routine to increase the number of levels in the data store ring buffer.
 * Calls to this routine after the data store has been initialized will have no effect.
 * The routine returns the new value of numDelayLevels
 */
int HyPerLayer::increaseDelayLevels(int neededDelay) {
   if (numDelayLevels < neededDelay + 1)
      numDelayLevels = neededDelay + 1;
   if (numDelayLevels > MAX_F_DELAY)
      numDelayLevels = MAX_F_DELAY;
   return numDelayLevels;
}

void HyPerLayer::requireMarginWidth(int marginWidthNeeded, int *marginWidthResult, char axis) {
   auto *layerGeometry = getComponentByType<LayerGeometry>();
   pvAssert(layerGeometry);
   layerGeometry->requireMarginWidth(marginWidthNeeded, axis);
   switch (axis) {
      case 'x':
         *marginWidthResult = getLayerLoc()->halo.lt;
         pvAssert(*marginWidthResult == getLayerLoc()->halo.rt);
         break;
      case 'y':
         *marginWidthResult = getLayerLoc()->halo.dn;
         pvAssert(*marginWidthResult == getLayerLoc()->halo.up);
         break;
      default: assert(0); break;
   }
   pvAssert(*marginWidthResult >= marginWidthNeeded);
}

/**
 * Returns the activity data for the layer.  This data is in the
 * extended space (with margins).
 */
const float *HyPerLayer::getLayerData(int delay) {
   PVLayerCube cube = publisher->createCube(delay);
   return cube.data;
}

Response::Status
HyPerLayer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ComponentBasedObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   if (mLayerInput) {
      mLayerInput->respond(message);
   }
   if (mInternalState) {
      mInternalState->respond(message);
   }
   if (mActivity) {
      mActivity->respond(message);
   }
   publisher->checkpointDataStore(checkpointer, getName(), "Delays");
   checkpointer->registerCheckpointData(
         std::string(getName()),
         std::string("lastUpdateTime"),
         &mLastUpdateTime,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   checkpointer->registerCheckpointData(
         std::string(getName()),
         std::string("nextWrite"),
         &writeTime,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);

   if (writeStep >= 0.0) {
      openOutputStateFile(message);
      if (sparseLayer) {
         checkpointer->registerCheckpointData(
               std::string(getName()),
               std::string("numframes_sparse"),
               &writeActivitySparseCalls,
               (std::size_t)1,
               true /*broadcast*/,
               false /*not constant*/);
      }
      else {
         checkpointer->registerCheckpointData(
               std::string(getName()),
               std::string("numframes"),
               &writeActivityCalls,
               (std::size_t)1,
               true /*broadcast*/,
               false /*not constant*/);
      }
   }

   // Timers

   update_timer = new Timer(getName(), "layer", "update ");
   checkpointer->registerTimer(update_timer);

#ifdef PV_USE_CUDA
   auto cudaDevice = mCudaDevice;
   if (cudaDevice) {
      gpu_update_timer = new PVCuda::CudaTimer(getName(), "layer", "gpuupdate");
      gpu_update_timer->setStream(cudaDevice->getStream());
      checkpointer->registerTimer(gpu_update_timer);
   }
#endif // PV_USE_CUDA

   publish_timer = new Timer(getName(), "layer", "publish");
   checkpointer->registerTimer(publish_timer);

   timescale_timer = new Timer(getName(), "layer", "timescale");
   checkpointer->registerTimer(timescale_timer);

   io_timer = new Timer(getName(), "layer", "io     ");
   checkpointer->registerTimer(io_timer);

   return Response::SUCCESS;
}

double HyPerLayer::getDeltaTriggerTime() const {
   if (triggerLayer != nullptr) {
      return triggerLayer->getDeltaUpdateTime();
   }
   else {
      return -1.0;
   }
}

bool HyPerLayer::needUpdate(double simTime, double dt) const {
   double deltaUpdateTime = getDeltaUpdateTime();
   if (deltaUpdateTime <= 0) {
      return false;
   }
   else if (triggerLayer != nullptr && triggerBehaviorType == UPDATEONLY_TRIGGER) {
      return triggerLayer->needUpdate(simTime + triggerOffset, dt);
   }
   else {
      double numUpdates    = (simTime - mLastUpdateTime) / deltaUpdateTime;
      double timeToClosest = std::fabs(numUpdates - std::nearbyint(numUpdates)) * deltaUpdateTime;
      return timeToClosest < 0.5 * dt;
   }
}

bool HyPerLayer::needReset(double simTime, double dt) {
   if (triggerLayer == nullptr) {
      return false;
   }
   if (triggerBehaviorType != RESETSTATE_TRIGGER) {
      return false;
   }
   if (getDeltaTriggerTime() <= 0) {
      return false;
   }
   if (simTime >= mLastTriggerTime + getDeltaTriggerTime()) {
      // TODO: test "simTime > mLastTriggerTime + getDeltaTriggerTime() - 0.5 * dt",
      // to avoid roundoff issues.
      return true;
   }
   return false;
}

Response::Status HyPerLayer::callUpdateState(double simTime, double dt) {
   auto status = Response::NO_ACTION;
   if (needReset(simTime, dt)) {
      resetStateOnTrigger();
      mLastTriggerTime = simTime;
   }

   update_timer->start();
#ifdef PV_USE_CUDA
   if (mUpdateGpu) {
      gpu_update_timer->start();
      assert(mUpdateGpu);
      status = updateStateGpu(simTime, dt);
      gpu_update_timer->stop();
   }
   else {
#endif
      status = updateState(simTime, dt);
#ifdef PV_USE_CUDA
   }
   // Activity updated, set flag to true
   updatedDeviceActivity  = true;
   updatedDeviceDatastore = true;
#endif
   update_timer->stop();
   mNeedToPublish  = true;
   mLastUpdateTime = simTime;
   return status;
}

void HyPerLayer::resetStateOnTrigger() {
   assert(triggerResetLayer != NULL);
   float *V = getV();
   if (V == NULL) {
      if (parent->getCommunicator()->commRank() == 0) {
         ErrorLog().printf(
               "%s: triggerBehavior is \"resetStateOnTrigger\" but layer does not have a membrane "
               "potential.\n",
               getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   InternalStateBuffer *resetVBuffer = triggerResetLayer->getComponentByType<InternalStateBuffer>();
   if (resetVBuffer != nullptr) {
      float const *resetV = resetVBuffer->getBufferData();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
      for (int k = 0; k < getNumNeuronsAllBatches(); k++) {
         V[k] = resetV[k];
      }
   }
   else {
      float const *resetA   = triggerResetLayer->getActivity();
      PVLayerLoc const *loc = triggerResetLayer->getLayerLoc();
      PVHalo const *halo    = &loc->halo;
      for (int b = 0; b < loc->nbatch; b++) {
         float const *resetABatch = resetA + (b * triggerResetLayer->getNumExtended());
         float *VBatch            = V + (b * triggerResetLayer->getNumNeurons());
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
         for (int k = 0; k < getNumNeurons(); k++) {
            int kex = kIndexExtended(
                  k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
            VBatch[k] = resetABatch[kex];
         }
      }
   }

   setActivity();

// Update V on GPU after CPU V gets set
#ifdef PV_USE_CUDA
   if (mUpdateGpu) {
      if (mInternalState) {
         mInternalState->copyToCuda();
      }
      // Right now, we're setting the activity on the CPU and memsetting the GPU memory
      // TODO calculate this on the GPU
      mActivity->copyToCuda();
      // We need to updateDeviceActivity and Datastore if we're resetting V
      updatedDeviceActivity  = true;
      updatedDeviceDatastore = true;
   }
#endif
}

#ifdef PV_USE_CUDA
int HyPerLayer::runUpdateKernel() {
   assert(mUpdateGpu);
   if (updatedDeviceGSyn) {
      if (mLayerInput->isUsingGPU()) { // (mLayerInput->isUsingGPU() || mUpdateGpu) {
         mLayerInput->copyToCuda();
      }
      updatedDeviceGSyn = false;
   }

   // V and Activity are write only buffers, so we don't need to do anything with them
   assert(krUpdate);

   // Sync all buffers before running
   syncGpu();

   // Run kernel
   krUpdate->run();

   return PV_SUCCESS;
}

Response::Status HyPerLayer::updateStateGpu(double timef, double dt) {
   Fatal() << "Update state for layer " << name << " is not implemented\n";
   return Response::NO_ACTION; // never reached; added to prevent compiler warnings.
}
#endif

Response::Status HyPerLayer::updateState(double timef, double dt) {
   // just copy accumulation buffer to membrane potential
   // and activity buffer (nonspiking)
   mInternalState->updateBuffer(timef, dt);
   const PVLayerLoc *loc = getLayerLoc();
   float *A              = getActivity();
   float *V              = getV();
   int num_channels      = getNumChannels();

   int nx          = loc->nx;
   int ny          = loc->ny;
   int nf          = loc->nf;
   int nbatch      = loc->nbatch;
   int num_neurons = nx * ny * nf;
   setActivity_HyPerLayer(
         nbatch,
         num_neurons,
         A,
         V,
         nx,
         ny,
         nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up);

   return Response::SUCCESS;
}

int HyPerLayer::setActivity() {
   const PVLayerLoc *loc = getLayerLoc();
   return setActivity_HyPerLayer(
         loc->nbatch,
         getNumNeurons(),
         mActivity->getActivity(),
         getV(),
         loc->nx,
         loc->ny,
         loc->nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up);
}

// Updates active indices for all levels (delays) here
void HyPerLayer::updateAllActiveIndices() { publisher->updateAllActiveIndices(); }

void HyPerLayer::updateActiveIndices() { publisher->updateActiveIndices(0); }

bool HyPerLayer::isExchangeFinished(int delay) { return publisher->isExchangeFinished(delay); }

bool HyPerLayer::isAllInputReady() {
   bool isReady = true;
   for (auto &c : recvConns) {
      auto *deliveryComponent = c->getComponentByType<BaseDelivery>();
      pvAssert(deliveryComponent);
      isReady &= deliveryComponent->isAllInputReady();
   }
   return isReady;
}

#ifdef PV_USE_CUDA
void HyPerLayer::syncGpu() {
   if (mLayerInput->isUsingGPU() || mUpdateGpu) {
      mCudaDevice->syncDevice();
   }
}
#endif

int HyPerLayer::publish(Communicator *comm, double simTime) {
   publish_timer->start();

   int status = PV_SUCCESS;
   if (mNeedToPublish) {
      if (mBoundaryConditions->getMirrorBCflag()) {
         mBoundaryConditions->applyBoundaryConditions(mActivity->getActivity(), getLayerLoc());
      }
      status         = publisher->publish(mLastUpdateTime);
      mNeedToPublish = false;
   }
   else {
      publisher->copyForward(mLastUpdateTime);
   }
   publish_timer->stop();
   return status;
}

int HyPerLayer::waitOnPublish(Communicator *comm) {
   publish_timer->start();

   // wait for MPI border transfers to complete
   //
   int status = publisher->wait();

   publish_timer->stop();
   return status;
}

/******************************************************************
 * FileIO
 *****************************************************************/

/* Inserts a new probe into an array of LayerProbes.
 *
 *
 *
 */
int HyPerLayer::insertProbe(LayerProbe *p) {
   if (p->getTargetLayer() != this) {
      WarnLog().printf(
            "HyPerLayer \"%s\": insertProbe called with probe %p, whose targetLayer is not this "
            "layer.  Probe was not inserted.\n",
            name,
            p);
      return numProbes;
   }
   for (int i = 0; i < numProbes; i++) {
      if (p == probes[i]) {
         WarnLog().printf(
               "HyPerLayer \"%s\": insertProbe called with probe %p, which has already been "
               "inserted as probe %d.\n",
               name,
               p,
               i);
         return numProbes;
      }
   }

   // malloc'ing a new buffer, copying data over, and freeing the old buffer could be replaced by
   // malloc
   LayerProbe **tmp;
   tmp = (LayerProbe **)malloc((numProbes + 1) * sizeof(LayerProbe *));
   assert(tmp != NULL);

   for (int i = 0; i < numProbes; i++) {
      tmp[i] = probes[i];
   }
   free(probes);

   probes            = tmp;
   probes[numProbes] = p;

   return ++numProbes;
}

Response::Status HyPerLayer::outputProbeParams() {
   for (int p = 0; p < numProbes; p++) {
      probes[p]->writeParams();
   }
   return Response::SUCCESS;
}

Response::Status HyPerLayer::outputState(double timestamp, double deltaTime) {
   io_timer->start();

   for (int i = 0; i < numProbes; i++) {
      probes[i]->outputStateWrapper(timestamp, deltaTime);
   }

   if (timestamp >= (writeTime - (deltaTime / 2)) && writeStep >= 0) {
      int writeStatus = PV_SUCCESS;
      writeTime += writeStep;
      if (sparseLayer) {
         writeStatus = writeActivitySparse(timestamp);
      }
      else {
         writeStatus = writeActivity(timestamp);
      }
      FatalIf(
            writeStatus != PV_SUCCESS,
            "%s: outputState failed on rank %d process.\n",
            getDescription_c(),
            parent->getCommunicator()->globalCommRank());
   }

   io_timer->stop();
   return Response::SUCCESS;
}

Response::Status HyPerLayer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   if (mInitializeFromCheckpointFlag) {
      readActivityFromCheckpoint(checkpointer);
      readDelaysFromCheckpoint(checkpointer);
      updateAllActiveIndices();
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}

void HyPerLayer::readActivityFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(std::string(name), std::string("A"), false);
}

void HyPerLayer::readDelaysFromCheckpoint(Checkpointer *checkpointer) {
   checkpointer->readNamedCheckpointEntry(std::string(name), std::string("Delays"), false);
}

Response::Status HyPerLayer::processCheckpointRead() {
   updateAllActiveIndices();
   return Response::SUCCESS;
}

int HyPerLayer::writeActivitySparse(double timed) {
   PVLayerCube cube      = publisher->createCube(0 /*delay*/);
   PVLayerLoc const *loc = getLayerLoc();
   pvAssert(cube.numItems == loc->nbatch * getNumExtended());

   int const mpiBatchDimension = getMPIBlock()->getBatchDimension();
   int const numFrames         = mpiBatchDimension * loc->nbatch;
   for (int frame = 0; frame < numFrames; frame++) {
      int const localBatchIndex = frame % loc->nbatch;
      int const mpiBatchIndex   = frame / loc->nbatch; // Integer division
      pvAssert(mpiBatchIndex * loc->nbatch + localBatchIndex == frame);

      SparseList<float> list;
      auto *activeIndicesBatch   = (SparseList<float>::Entry const *)cube.activeIndices;
      auto *activeIndicesElement = &activeIndicesBatch[localBatchIndex * getNumExtended()];
      PVLayerLoc const *loc      = getLayerLoc();
      int nxExt                  = loc->nx + loc->halo.lt + loc->halo.rt;
      int nyExt                  = loc->ny + loc->halo.dn + loc->halo.up;
      int nf                     = loc->nf;
      for (long int k = 0; k < cube.numActive[localBatchIndex]; k++) {
         SparseList<float>::Entry entry = activeIndicesElement[k];
         int index                      = (int)entry.index;

         // Location is local extended; need global restricted.
         // Get local restricted coordinates.
         int x = kxPos(index, nxExt, nyExt, nf) - loc->halo.lt;
         if (x < 0 or x >= loc->nx) {
            continue;
         }
         int y = kyPos(index, nxExt, nyExt, nf) - loc->halo.up;
         if (y < 0 or y >= loc->ny) {
            continue;
         }
         // Convert to global restricted coordinates.
         x += loc->kx0;
         y += loc->ky0;
         int f = featureIndex(index, nxExt, nyExt, nf);

         // Get global restricted index.
         entry.index = (uint32_t)kIndex(x, y, f, loc->nxGlobal, loc->nyGlobal, nf);
         list.addEntry(entry);
      }
      auto gatheredList =
            BufferUtils::gatherSparse(getMPIBlock(), list, mpiBatchIndex, 0 /*root process*/);
      if (getMPIBlock()->getRank() == 0) {
         long fpos = mOutputStateStream->getOutPos();
         if (fpos == 0L) {
            BufferUtils::ActivityHeader header = BufferUtils::buildSparseActivityHeader<float>(
                  loc->nx * getMPIBlock()->getNumColumns(),
                  loc->ny * getMPIBlock()->getNumRows(),
                  loc->nf,
                  0 /* numBands */); // numBands will be set by call to incrementNBands.
            header.timestamp = timed;
            BufferUtils::writeActivityHeader(*mOutputStateStream, header);
         }
         BufferUtils::writeSparseFrame(*mOutputStateStream, &gatheredList, timed);
      }
   }
   writeActivitySparseCalls += numFrames;
   updateNBands(writeActivitySparseCalls);
   return PV_SUCCESS;
}

// write non-spiking activity
int HyPerLayer::writeActivity(double timed) {
   PVLayerCube cube      = publisher->createCube(0);
   PVLayerLoc const *loc = getLayerLoc();
   pvAssert(cube.numItems == loc->nbatch * getNumExtended());

   PVHalo const &halo   = loc->halo;
   int const nxExtLocal = loc->nx + halo.lt + halo.rt;
   int const nyExtLocal = loc->ny + halo.dn + halo.up;
   int const nf         = loc->nf;

   int const mpiBatchDimension = getMPIBlock()->getBatchDimension();
   int const numFrames         = mpiBatchDimension * loc->nbatch;
   for (int frame = 0; frame < numFrames; frame++) {
      int const localBatchIndex = frame % loc->nbatch;
      int const mpiBatchIndex   = frame / loc->nbatch; // Integer division
      pvAssert(mpiBatchIndex * loc->nbatch + localBatchIndex == frame);

      float *data = &cube.data[localBatchIndex * getNumExtended()];
      Buffer<float> localBuffer(data, nxExtLocal, nyExtLocal, nf);
      localBuffer.crop(loc->nx, loc->ny, Buffer<float>::CENTER);
      Buffer<float> blockBuffer = BufferUtils::gather<float>(
            getMPIBlock(), localBuffer, loc->nx, loc->ny, mpiBatchIndex, 0 /*root process*/);
      // At this point, the rank-zero process has the entire block for the batch element,
      // regardless of what the mpiBatchIndex is.
      if (getMPIBlock()->getRank() == 0) {
         long fpos = mOutputStateStream->getOutPos();
         if (fpos == 0L) {
            BufferUtils::ActivityHeader header = BufferUtils::buildActivityHeader<float>(
                  loc->nx * getMPIBlock()->getNumColumns(),
                  loc->ny * getMPIBlock()->getNumRows(),
                  loc->nf,
                  0 /* numBands */); // numBands will be set by call to incrementNBands.
            header.timestamp = timed;
            BufferUtils::writeActivityHeader(*mOutputStateStream, header);
         }
         BufferUtils::writeFrame<float>(*mOutputStateStream, &blockBuffer, timed);
      }
   }
   writeActivityCalls += numFrames;
   updateNBands(writeActivityCalls);
   return PV_SUCCESS;
}

void HyPerLayer::updateNBands(int const numCalls) {
   // Only the root process needs to maintain INDEX_NBANDS, so only the root process modifies
   // numCalls
   // This way, writeActivityCalls does not need to be coordinated across MPI
   if (mOutputStateStream != nullptr) {
      long int fpos = mOutputStateStream->getOutPos();
      mOutputStateStream->setOutPos(sizeof(int) * INDEX_NBANDS, true /*fromBeginning*/);
      mOutputStateStream->write(&numCalls, (long)sizeof(numCalls));
      mOutputStateStream->setOutPos(fpos, true /*fromBeginning*/);
   }
}

bool HyPerLayer::localDimensionsEqual(PVLayerLoc const *loc1, PVLayerLoc const *loc2) {
   return loc1->nbatch == loc2->nbatch && loc1->nx == loc2->nx && loc1->ny == loc2->ny
          && loc1->nf == loc2->nf && loc1->halo.lt == loc2->halo.lt
          && loc1->halo.rt == loc2->halo.rt && loc1->halo.dn == loc2->halo.dn
          && loc1->halo.up == loc2->halo.up;
}

} // end of PV namespace
