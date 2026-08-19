/*
 * BaseLayer.cpp
 *
 *  The top of the hierarchy for layer classes.
*/

#include "BaseLayer.hpp"
#include "components/HyPerActivityBuffer.hpp"
#include "components/PublisherComponent.hpp"
#include "observerpattern/ObserverTable.hpp"
#include "utils/BufferUtilsMPI.hpp"

namespace PV {

// This constructor is protected so that only derived classes can call it.
// When a layer is derived from another layer, its constructor should call the base layer's
// constructor without any arguments (the default constructor) The derived class's constructor
// should call an initialize() function member that calls the base class's initialize() function.
// Generally, a layer's default constructor should be protected and do nothing; in any
// event it should NOT call any virtual methods, because the base class's method will be
// called instead of the derived class's overriding method.
BaseLayer::BaseLayer() {}

BaseLayer::~BaseLayer() {}

///////
/// Classes derived from BaseLayer should have an initialize() function member that
/// calls BaseLayer::initialize(), and the derived class's constructor should call
/// its initialize(). This way, BaseLayer::initialize() can call virtual methods and
/// the derived class's method will be the one that gets called.
void BaseLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   ComponentBasedObject::initialize(name, params, comm);

   // The layer writes this flag to output params file. ParamsInterface-derived components of the
   // layer will automatically read InitializeFromCheckpointFlag, but shouldn't also write it.
   mWriteInitializeFromCheckpointFlag = true;
}

/******************************************************************
 * Define actions for layer-specific messages
 *****************************************************************/
void BaseLayer::initMessageActionMap() {
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
void BaseLayer::fillComponentTable() {
   Subject::fillComponentTable();
   mLayerGeometry = createLayerGeometry();
   if (mLayerGeometry) {
      addUniqueComponent(mLayerGeometry);
   }
   mPhaseParam = createPhaseParam();
   if (mPhaseParam) {
      addUniqueComponent(mPhaseParam);
   }
   mLayerUpdateController = createLayerUpdateController();
   if (mLayerUpdateController) {
      addUniqueComponent(mLayerUpdateController);
   }
   mActivityComponent = createActivityComponent();
   if (mActivityComponent) {
      addUniqueComponent(mActivityComponent);
   }
   mPublisher = createPublisher();
   if (mPublisher) {
      addUniqueComponent(mPublisher);
   }
   mBoundaryConditions = createBoundaryConditions();
   if (mBoundaryConditions) {
      addUniqueComponent(mBoundaryConditions);
   }
   mLayerOutput = createLayerOutput();
   if (mLayerOutput) {
      addUniqueComponent(mLayerOutput);
   }
}

LayerGeometry *BaseLayer::createLayerGeometry() {
   return new LayerGeometry(getName(), parameters(), mCommunicator);
}

PhaseParam *BaseLayer::createPhaseParam() {
   return new PhaseParam(getName(), parameters(), mCommunicator);
}

LayerUpdateController *BaseLayer::createLayerUpdateController() {
   return new LayerUpdateController(getName(), parameters(), mCommunicator);
}

ActivityComponent *BaseLayer::createActivityComponent() {
   return new ActivityComponent(getName(), parameters(), mCommunicator);
}

BasePublisherComponent *BaseLayer::createPublisher() {
   return new PublisherComponent(getName(), parameters(), mCommunicator);
}

BoundaryConditions *BaseLayer::createBoundaryConditions() {
   return new BoundaryConditions(getName(), parameters(), mCommunicator);
}

LayerOutputComponent *BaseLayer::createLayerOutput() {
   return new LayerOutputComponent(getName(), parameters(), mCommunicator);
}

/******************************************************************
 * Read/write params for layer and for layer components
 *****************************************************************/
int BaseLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ComponentBasedObject::ioParamsFillGroup(ioFlag);
   return status;
}

/******************************************************************
 * CommunicateInitInfo stage
 *****************************************************************/
Response::Status
BaseLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
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
#endif

   return Response::SUCCESS;
}

void BaseLayer::synchronizeMarginWidth(BaseLayer *layer) {
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

/******************************************************************
 * AllocateDataStructures stage
 *****************************************************************/
Response::Status BaseLayer::allocateDataStructures() {
   // If not mirroring, fill the boundaries with the value in the valueBC param
   Response::Status status = ComponentBasedObject::allocateDataStructures();
   if (!mBoundaryConditions) { return status; }
   if (!mBoundaryConditions->getMirrorBCflag() && mBoundaryConditions->getValueBC() != 0.0f) {
      auto *activityBuffer = mActivityComponent->getComponentByType<ActivityBuffer>();
      auto *activityData   = activityBuffer->getReadWritePointer();
      mBoundaryConditions->applyBoundaryConditions(activityData, getLayerLoc());
      status = Response::SUCCESS;
   }
   return status;
}

Response::Status
BaseLayer::respondLayerWriteParams(std::shared_ptr<LayerWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status BaseLayer::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   Response::Status status = ComponentBasedObject::setCudaDevice(message);
   if (Response::completed(status)) {
      status = notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   }
   return status;
}
#endif // PV_USE_CUDA

Response::Status
BaseLayer::respondLayerSetMaxPhase(std::shared_ptr<LayerSetMaxPhaseMessage const> message) {
   return notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
}

/******************************************************************
 * RegisterData stage
 *****************************************************************/
Response::Status
BaseLayer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
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
BaseLayer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   if (mLayerUpdateController) {
      mLayerUpdateController->respond(message);
   }
   if (mActivityComponent) {
      InfoLog().printf("Initializing state of %s\n", getDescription_c());
      mActivityComponent->respond(message);
   }
   return Response::SUCCESS;
}

Response::Status BaseLayer::respondLayerClearProgressFlags(
      std::shared_ptr<LayerClearProgressFlagsMessage const> message) {
   if (mLayerUpdateController) {
      mLayerUpdateController->respond(message);
   }
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status
BaseLayer::respondLayerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) {
   if (message->mPhase != mPhaseParam->getPhase()) {
      return Response::NO_ACTION;
   }
   return layerCopyFromGpu(message);
}

Response::Status
BaseLayer::layerCopyFromGpu(std::shared_ptr<LayerCopyFromGpuMessage const> message) {
   if (mActivityComponent and mActivityComponent->isUsingGPU()) {
      mActivityComponent->copyFromCuda();
   }
   return Response::SUCCESS;
}


Response::Status BaseLayer::copyInitialStateToGPU() {
   if (mActivityComponent) {
      return mActivityComponent->respond(std::make_shared<CopyInitialStateToGPUMessage>());
   }
   else {
      return Response::NO_ACTION;
   }
}
#endif // PV_USE_CUDA

/******************************************************************
 * Run-loop stage
 *****************************************************************/
Response::Status BaseLayer::respondLayerAdvanceDataStore(
      std::shared_ptr<LayerAdvanceDataStoreMessage const> message) {
   if (message->mPhase < 0 || message->mPhase == mPhaseParam->getPhase()) {
      mPublisher->respond(message);
   }
   return Response::SUCCESS;
}

Response::Status
BaseLayer::respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message) {
   if (message->mPhase != mPhaseParam->getPhase()) {
      return Response::NO_ACTION;
   }
   mPublisher->respond(message);
   return Response::SUCCESS;
}

Response::Status
BaseLayer::respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   auto status = Response::NO_ACTION;
   if (mLayerOutput and message->mPhase == mPhaseParam->getPhase()) {
      status = mLayerOutput->respond(message);
   }
   return status;
}

Response::Status BaseLayer::respondLayerRecvSynapticInput(
      std::shared_ptr<LayerRecvSynapticInputMessage const> message) {
   if (mLayerUpdateController) {
      mLayerUpdateController->respond(message);
   }
   return Response::SUCCESS;
}

Response::Status
BaseLayer::respondLayerUpdateState(std::shared_ptr<LayerUpdateStateMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (mLayerUpdateController) {
      status = mLayerUpdateController->respond(message);
   }
   checkUpdateState(message->mTime, message->mDeltaT);
   return status;
}

Response::Status BaseLayer::checkUpdateState(double simTime, double deltaTime) {
   return Response::NO_ACTION;
}

Response::Status BaseLayer::respondLayerCheckNotANumber(
      std::shared_ptr<LayerCheckNotANumberMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (message->mPhase != mPhaseParam->getPhase()) {
      return status;
   }
   auto layerData = mPublisher->getLayerData();
   long const N   = getNumExtendedAllBatches();
   for (long n = 0; n < N; n++) {
      float a = layerData[n];
      FatalIf(
            a != a,
            "%s has not-a-number values in the activity buffer.  Exiting.\n",
            getDescription_c());
   }
   return status;
}

} // end of PV namespace
