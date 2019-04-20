/*
 * HyPerLayer.cpp
 *
 *  Created on: Jul 29, 2008
 *
 *  The top of the hierarchy for layer classes.
*/

#include "HyPerLayer.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "components/HyPerActivityComponent.hpp"
#include "components/HyPerInternalStateBuffer.hpp"
#include "components/PublisherComponent.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "utils/BufferUtilsMPI.hpp"

namespace PV {

// This constructor is protected so that only derived classes can call it.
// It should be called as the normal method of object construction by
// derived classes.  It should NOT call any virtual methods
HyPerLayer::HyPerLayer() {}

HyPerLayer::HyPerLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerLayer::~HyPerLayer() {}

///////
/// Classes derived from HyPerLayer should call HyPerLayer::initialize themselves
/// to take advantage of virtual methods.  Note that the HyPerLayer constructor
/// does not call initialize.  This way, HyPerLayer::initialize can call virtual
/// methods and the derived class's method will be the one that gets called.
void HyPerLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   ComponentBasedObject::initialize(name, params, comm);

   // The layer writes this flag to output params file. ParamsInterface-derived components of the
   // layer will automatically read InitializeFromCheckpointFlag, but shouldn't also write it.
   mWriteInitializeFromCheckpointFlag = true;
}

/******************************************************************
 * Define actions for layer-specific messages
 *****************************************************************/
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

/******************************************************************
 * Create components
 *****************************************************************/
void HyPerLayer::fillComponentTable() {
   Subject::fillComponentTable();
   mLayerGeometry = createLayerGeometry();
   if (mLayerGeometry) {
      addUniqueComponent(mLayerGeometry);
   }
   mPhaseParam = createPhaseParam();
   if (mPhaseParam) {
      addUniqueComponent(mPhaseParam);
   }
   mBoundaryConditions = createBoundaryConditions();
   if (mBoundaryConditions) {
      addUniqueComponent(mBoundaryConditions);
   }
   mLayerUpdateController = createLayerUpdateController();
   if (mLayerUpdateController) {
      addUniqueComponent(mLayerUpdateController);
   }
   mLayerInput = createLayerInput();
   if (mLayerInput) {
      addUniqueComponent(mLayerInput);
   }
   mActivityComponent = createActivityComponent();
   if (mActivityComponent) {
      addUniqueComponent(mActivityComponent);
   }
   mPublisher = createPublisher();
   if (mPublisher) {
      addUniqueComponent(mPublisher);
   }
   mLayerOutput = createLayerOutput();
   if (mLayerOutput) {
      addUniqueComponent(mLayerOutput);
   }
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

LayerUpdateController *HyPerLayer::createLayerUpdateController() {
   return new LayerUpdateController(name, parameters(), mCommunicator);
}

LayerInputBuffer *HyPerLayer::createLayerInput() {
   return new LayerInputBuffer(name, parameters(), mCommunicator);
}

ActivityComponent *HyPerLayer::createActivityComponent() {
   return new HyPerActivityComponent<GSynAccumulator,
                                     HyPerInternalStateBuffer,
                                     HyPerActivityBuffer>(name, parameters(), mCommunicator);
}

BasePublisherComponent *HyPerLayer::createPublisher() {
   return new PublisherComponent(name, parameters(), mCommunicator);
}

LayerOutputComponent *HyPerLayer::createLayerOutput() {
   return new LayerOutputComponent(name, parameters(), mCommunicator);
}

/******************************************************************
 * Read/write params for layer and for layer components
 *****************************************************************/
int HyPerLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ComponentBasedObject::ioParamsFillGroup(ioFlag);
   ioParam_dataType(ioFlag);
   return status;
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

/******************************************************************
 * CommunicateInitInfo stage
 *****************************************************************/
Response::Status
HyPerLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = ComponentBasedObject::communicateInitInfo(message);
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

   return Response::SUCCESS;
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

Response::Status
HyPerLayer::respondLayerWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status HyPerLayer::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   Response::Status status = ComponentBasedObject::setCudaDevice(message);
   if (Response::completed(status)) {
      status = notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   }
   return status;
}
#endif // PV_USE_CUDA

/******************************************************************
 * AllocateDataStructures stage
 *****************************************************************/
Response::Status HyPerLayer::allocateDataStructures() {
   // If not mirroring, fill the boundaries with the value in the valueBC param
   Response::Status status = ComponentBasedObject::allocateDataStructures();
   if (!mBoundaryConditions->getMirrorBCflag() && mBoundaryConditions->getValueBC() != 0.0f) {
      auto *activityBuffer = mActivityComponent->getComponentByType<ActivityBuffer>();
      auto *activityData   = activityBuffer->getReadWritePointer();
      mBoundaryConditions->applyBoundaryConditions(activityData, getLayerLoc());
      status = Response::SUCCESS;
   }
   return status;
}

Response::Status
HyPerLayer::respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message) {
   return notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
}

/******************************************************************
 * RegisterData stage
 *****************************************************************/
Response::Status
HyPerLayer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ComponentBasedObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }

   return Response::SUCCESS;
}

/******************************************************************
 * InitializeState stage
 *****************************************************************/
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
Response::Status HyPerLayer::respondLayerClearProgressFlags(
      std::shared_ptr<LayerClearProgressFlagsMessage const> message) {
   if (mLayerUpdateController) {
      mLayerUpdateController->respond(message);
   }
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status
HyPerLayer::respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != mPhaseParam->getPhase()) {
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

Response::Status HyPerLayer::copyInitialStateToGPU() {
   return mActivityComponent->respond(std::make_shared<CopyInitialStateToGPUMessage>());
}
#endif // PV_USE_CUDA

/******************************************************************
 * Run-loop stage
 *****************************************************************/
Response::Status HyPerLayer::respondLayerAdvanceDataStore(
      std::shared_ptr<LayerAdvanceDataStoreMessage const> message) {
   if (message->mPhase < 0 || message->mPhase == mPhaseParam->getPhase()) {
      mPublisher->respond(message);
   }
   return Response::SUCCESS;
}

Response::Status
HyPerLayer::respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message) {
   if (message->mPhase != mPhaseParam->getPhase()) {
      return Response::NO_ACTION;
   }
   mPublisher->publish(mCommunicator, message->mTime);
   return Response::SUCCESS;
}

Response::Status
HyPerLayer::respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   auto status = Response::NO_ACTION;
   if (mLayerOutput and message->mPhase == mPhaseParam->getPhase()) {
      status = mLayerOutput->respond(message);
   }
   return status;
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
      status = mLayerUpdateController->respond(message);
   }
   checkUpdateState(message->mTime, message->mDeltaT);
   return status;
}

Response::Status HyPerLayer::checkUpdateState(double simTime, double deltaTime) {
   return Response::NO_ACTION;
}

Response::Status HyPerLayer::respondLayerCheckNotANumber(
      std::shared_ptr<LayerCheckNotANumberMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != mPhaseParam->getPhase()) {
      return status;
   }
   auto layerData = mPublisher->getLayerData();
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

} // end of PV namespace
