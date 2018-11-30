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
#include "components/HyPerActivityBuffer.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "connections/BaseConnection.hpp"
#include "include/default_params.h"
#include "include/pv_common.h"
#include "io/FileStream.hpp"
#include "io/io.hpp"
#include "observerpattern/ObserverTable.hpp"
#include <assert.h>
#include <iostream>
#include <sstream>
#include <string.h>

namespace PV {

// This constructor is protected so that only derived classes can call it.
// It should be called as the normal method of object construction by
// derived classes.  It should NOT call any virtual methods
HyPerLayer::HyPerLayer() { initialize_base(); }

HyPerLayer::HyPerLayer(const char *name, PVParams *params, Communicator *comm) {
   initialize_base();
   initialize(name, params, comm);
}

// initialize_base should be called only by constructors.  It should not
// call any virtual methods, because polymorphism is not available when
// a base class constructor is inherited from a derived class constructor.
// In general, initialize_base should be used only to initialize member variables
// to safe values.
int HyPerLayer::initialize_base() {
   name             = NULL;
   probes           = NULL;
   numProbes        = 0;
   writeTime        = 0;
   initialWriteTime = 0;

#ifdef PV_USE_CUDA
   allocDeviceDatastore     = false;
   allocDeviceActiveIndices = false;
   d_Datastore              = NULL;
   d_ActiveIndices          = NULL;
   d_numActive              = NULL;
   updatedDeviceDatastore   = true;
#ifdef PV_USE_CUDNN
   cudnn_Datastore = NULL;
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

   publish_timer = NULL;
   io_timer      = NULL;

   recvConns.clear();

   return PV_SUCCESS;
}

///////
/// Classes derived from HyPerLayer should call HyPerLayer::initialize themselves
/// to take advantage of virtual methods.  Note that the HyPerLayer constructor
/// does not call initialize.  This way, HyPerLayer::initialize can call virtual
/// methods and the derived class's method will be the one that gets called.
void HyPerLayer::initialize(const char *name, PVParams *params, Communicator *comm) {
   ComponentBasedObject::initialize(name, params, comm);

   // The layer writes this flag to output params file. ParamsInterface-derived components of the
   // layer will automatically read InitializeFromCheckpointFlag, but shouldn't also write it.
   mWriteInitializeFromCheckpointFlag = true;

   writeTime                = initialWriteTime;
   writeActivityCalls       = 0;
   writeActivitySparseCalls = 0;
   numDelayLevels = 1; // If a connection has positive delay so that more delay levels are needed,
   // numDelayLevels is increased when BaseConnection::communicateInitInfo calls
   // increaseDelayLevels
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

#ifdef PV_USE_CUDA
   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerCopyFromGpuMessage const>(msgptr);
      return respondLayerCopyFromGpu(castMessage);
   };
   mMessageActionMap.emplace("LayerCopyFromGpu", action);
#endif // PV_USE_CUDA

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

void HyPerLayer::createComponentTable(char const *description) {
   pvAssert(mTable == nullptr);
   Subject::createComponentTable(description);
   pvAssert(mTable != nullptr);
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
   mLayerUpdateController = createLayerUpdateController();
   if (mLayerUpdateController) {
      addUniqueComponent(mLayerUpdateController->getDescription(), mLayerUpdateController);
   }
   mLayerInput = createLayerInput();
   if (mLayerInput) {
      addUniqueComponent(mLayerInput->getDescription(), mLayerInput);
   }
   mActivityComponent = createActivityComponent();
   if (mActivityComponent) {
      addUniqueComponent(mActivityComponent->getDescription(), mActivityComponent);
   }
}

LayerUpdateController *HyPerLayer::createLayerUpdateController() {
   return new LayerUpdateController(name, parameters(), mCommunicator);
}

LayerGeometry *HyPerLayer::createLayerGeometry() {
   return new LayerGeometry(name, parameters(), mCommunicator);
}

PhaseParam *HyPerLayer::createPhaseParam() {
   return new PhaseParam(name, parameters(), mCommunicator);
}

BoundaryConditions *HyPerLayer::createBoundaryConditions() {
   return new BoundaryConditions(name, parameters(), mCommunicator);
}

LayerInputBuffer *HyPerLayer::createLayerInput() {
   return new LayerInputBuffer(name, parameters(), mCommunicator);
}

ActivityComponent *HyPerLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     HyPerActivityBuffer>(name, parameters(), mCommunicator);
}

HyPerLayer::~HyPerLayer() {
   delete publish_timer;
   delete io_timer;

   delete mOutputStateStream;

#ifdef PV_USE_CUDA
   if (d_Datastore) {
      delete d_Datastore;
   }

#ifdef PV_USE_CUDNN
   if (cudnn_Datastore) {
      delete cudnn_Datastore;
   }
#endif // PV_USE_CUDNN
#endif // PV_USE_CUDA

   free(probes); // All probes are deleted by the HyPerCol, so probes[i] doesn't need to be deleted,
   // only the array itself.

   delete publisher;
}

void HyPerLayer::addPublisher() {
   MPIBlock const *mpiBlock  = mCommunicator->getLocalMPIBlock();
   PVLayerCube *activityCube = (PVLayerCube *)malloc(sizeof(PVLayerCube));
   activityCube->numItems    = getNumExtendedAllBatches();
   auto *activityBuffer      = mActivityComponent->getComponentByType<ActivityBuffer>();
   activityCube->data        = activityBuffer->getBufferData();
   activityCube->loc         = *getLayerLoc();
   publisher = new Publisher(*mpiBlock, activityCube, getNumDelayLevels(), getSparseFlag());
}

Response::Status
HyPerLayer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   if (mLayerUpdateController) {
      mLayerUpdateController->respond(message);
   }
   if (mActivityComponent) {
      mActivityComponent->respond(message);
   }
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status HyPerLayer::copyInitialStateToGPU() {
   return mActivityComponent->respond(std::make_shared<CopyInitialStateToGPUMessage>());
}
#endif // PV_USE_CUDA

int HyPerLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   // Derived classes with new params behavior should override ioParamsFillGroup
   // and the overriding method should call the base class's ioParamsFillGroup.
   for (auto *c : *mTable) {
      auto obj = dynamic_cast<BaseObject *>(c);
      if (obj) {
         obj->ioParams(ioFlag, false, false);
      }
   }
   ioParam_writeStep(ioFlag);
   ioParam_initialWriteTime(ioFlag);
   ioParam_sparseLayer(ioFlag);

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
      if (mCommunicator->globalCommRank() == 0) {
         WarnLog().printf(
               "%s defines the dataType param, which is no longer used.\n", getDescription_c());
      }
   }
}

void HyPerLayer::ioParam_writeStep(enum ParamsIOFlag ioFlag) {
   bool warnIfAbsent = false; // If not in params, will be set in CommunicateInitInfo stage
   // If writing a derived class that overrides ioParam_writeStep, check if the setDefaultWriteStep
   // method also needs to be overridden.
   parameters()->ioParamValue(ioFlag, name, "writeStep", &writeStep, writeStep, warnIfAbsent);
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
         if (mCommunicator->globalCommRank() == 0) {
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
   return notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
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
   if (mLayerUpdateController) {
      mLayerUpdateController->respond(message);
   }
   return Response::SUCCESS;
}

Response::Status HyPerLayer::respondLayerRecvSynapticInput(
      std::shared_ptr<LayerRecvSynapticInputMessage const> message) {
   if (mLayerUpdateController) {
      mLayerUpdateController->respond(message);
   }
   return Response::SUCCESS;
}

Response::Status
HyPerLayer::respondLayerUpdateState(std::shared_ptr<LayerUpdateStateMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (mLayerUpdateController) {
      mLayerUpdateController->respond(message);
   }
   checkUpdateState(message->mTime, message->mDeltaT);
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status
HyPerLayer::respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != getPhase()) {
      return status;
   }
   message->mTimer->start();
   if (mActivityComponent and mActivityComponent->isUsingGPU()) {
      mActivityComponent->copyFromCuda();
   }
   if (mLayerInput and mLayerInput->isUsingGPU()) {
      mLayerInput->respond(message);
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
   publish(mCommunicator, message->mTime);
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

#ifdef PV_USE_CUDA

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

   return status;
}

Response::Status HyPerLayer::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   Response::Status status = ComponentBasedObject::setCudaDevice(message);
   return notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
}

#endif // PV_USE_CUDA

Response::Status
HyPerLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *hierarchy      = message->mHierarchy;
   auto *tableComponent = mTable->lookupByType<ObserverTable>();
   if (!tableComponent) {
      std::string tableDescription = std::string("ObserverTable \"") + getName() + "\"";
      tableComponent               = new ObserverTable(tableDescription.c_str());
      tableComponent->copyTable(hierarchy);
      addUniqueComponent(tableComponent->getDescription(), tableComponent);
      // mTable takes ownership of tableComponent, which will therefore be deleted by the
      // Subject::deleteObserverTable() method during destructor.
   }
   pvAssert(tableComponent);

   auto communicateMessage = std::make_shared<CommunicateInitInfoMessage>(
         mTable,
         message->mDeltaTime,
         message->mNxGlobal,
         message->mNyGlobal,
         message->mNBatchGlobal,
         message->mNumThreads);

   Response::Status status =
         notify(communicateMessage, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   if (!Response::completed(status)) {
      return status;
   }
#ifdef PV_USE_CUDA
   // Set UsingGPUFlag if any of the components use GPU.
   for (auto *c : *mTable) {
      auto obj = dynamic_cast<BaseObject *>(c);
      if (obj) {
         mUsingGPUFlag |= obj->isUsingGPU();
      }
   }

   // Here, the connection tells all participating recv layers to allocate memory on gpu
   // if receive from gpu is set. These buffers should be set in allocate
   if (mActivityComponent->getUpdateGpu()) {
      if (mLayerInput) {
         mLayerInput->useCuda();
      }
   }
#endif

   if (!parameters()->present(getName(), "writeStep")) {
      setDefaultWriteStep(message);
   }

   return Response::SUCCESS;
}

void HyPerLayer::setDefaultWriteStep(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   writeStep = message->mDeltaTime;
   // Call ioParamValue to generate the warnIfAbsent warning.
   parameters()->ioParamValue(PARAMS_IO_READ, name, "writeStep", &writeStep, writeStep, true);
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

Response::Status HyPerLayer::allocateDataStructures() {
   // Once initialize and communicateInitInfo have been called, HyPerLayer has the
   // information it needs to allocate the membrane potential buffer V, the
   // activity buffer activity->data, and the data store.
   auto status = Response::SUCCESS;

   auto allocateMessage = std::make_shared<AllocateDataStructuresMessage>();
   notify(allocateMessage, mCommunicator->globalCommRank() == 0 /*printFlag*/);

   const PVLayerLoc *loc = getLayerLoc();
   int nx                = loc->nx;
   int ny                = loc->ny;
   int nf                = loc->nf;
   PVHalo const *halo    = &loc->halo;

   // If not mirroring, fill the boundaries with the value in the valueBC param
   if (!mBoundaryConditions->getMirrorBCflag() && mBoundaryConditions->getValueBC() != 0.0f) {
      auto *activityBuffer = mActivityComponent->getComponentByType<ActivityBuffer>();
      auto *activityData   = activityBuffer->getReadWritePointer();
      mBoundaryConditions->applyBoundaryConditions(activityData, getLayerLoc());
   }

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
            mCommunicator->globalCommRank(),
            strerror(errno));
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
   notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   auto *checkpointer = message->mDataRegistry;
   publisher->checkpointDataStore(checkpointer, getName(), "Delays");
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

   publish_timer = new Timer(getName(), "layer", "publish");
   checkpointer->registerTimer(publish_timer);

   io_timer = new Timer(getName(), "layer", "io     ");
   checkpointer->registerTimer(io_timer);

   return Response::SUCCESS;
}

Response::Status HyPerLayer::checkUpdateState(double simTime, double deltaTime) {
   return Response::NO_ACTION;
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

int HyPerLayer::publish(Communicator *comm, double simTime) {
   publish_timer->start();

   int status = PV_SUCCESS;
   double lastUpdateTime =
         mLayerUpdateController ? mLayerUpdateController->getLastUpdateTime() : 0.0;
   if (lastUpdateTime >= simTime) {
      if (mBoundaryConditions->getMirrorBCflag()) {
         auto *activityBuffer = mActivityComponent->getComponentByType<ActivityBuffer>();
         auto *activityData   = activityBuffer->getReadWritePointer();
         mBoundaryConditions->applyBoundaryConditions(activityData, getLayerLoc());
      }
      status = publisher->publish(lastUpdateTime);
#ifdef PV_USE_CUDA
      setUpdatedDeviceDatastoreFlag(true);
#endif // PV_USE_CUDA
   }
   else {
      publisher->copyForward(lastUpdateTime);
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
            mCommunicator->globalCommRank());
   }

   io_timer->stop();
   return Response::SUCCESS;
}

Response::Status HyPerLayer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   pvAssert(mInitializeFromCheckpointFlag);
   auto status = Response::NO_ACTION;
   status      = ComponentBasedObject::readStateFromCheckpoint(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   readDelaysFromCheckpoint(checkpointer);
   updateAllActiveIndices();
   return Response::SUCCESS;
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

      float const *data = &cube.data[localBatchIndex * getNumExtended()];
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

} // end of PV namespace
