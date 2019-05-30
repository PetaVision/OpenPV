/*
 * BasePublisherComponent.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: peteschultz
 */

#include "BasePublisherComponent.hpp"

namespace PV {

BasePublisherComponent::BasePublisherComponent(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

BasePublisherComponent::BasePublisherComponent() {}

BasePublisherComponent::~BasePublisherComponent() {
   delete mPublishTimer;
   delete mPublisher;
}

void BasePublisherComponent::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void BasePublisherComponent::setObjectType() { mObjectType = "BasePublisherComponent"; }

void BasePublisherComponent::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

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
}

Response::Status BasePublisherComponent::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *objectTable = message->mObjectTable;
   if (mActivity == nullptr) {
      mActivity = objectTable->findObject<ActivityBuffer>(getName());
   }
   FatalIf(!mActivity, "%s requires an ActivityBuffer component.\n", getDescription_c());
   if (mBoundaryConditions == nullptr) {
      mBoundaryConditions = objectTable->findObject<BoundaryConditions>(getName());
   }
   // It is not an error for BoundaryConditions to be null.
   if (mUpdateController == nullptr) {
      mUpdateController = objectTable->findObject<LayerUpdateController>(getName());
   }
   // It is not an error for UpdateController to be null.
   return Response::SUCCESS;
}

int BasePublisherComponent::increaseDelayLevels(int neededDelay) {
   if (mNumDelayLevels < neededDelay + 1)
      mNumDelayLevels = neededDelay + 1;
   if (mNumDelayLevels > MAX_F_DELAY)
      mNumDelayLevels = MAX_F_DELAY;
   return mNumDelayLevels;
}

Response::Status BasePublisherComponent::allocateDataStructures() {
   if (!mActivity->getDataStructuresAllocatedFlag()) {
      return Response::POSTPONE;
   }
   mPublisher = new Publisher(
         *mCommunicator->getLocalMPIBlock(),
         mActivity->getBufferData(),
         mActivity->getLayerLoc(),
         getNumDelayLevels(),
         mSparseLayer);
#ifdef PV_USE_CUDA
   allocateCudaBuffers();
#endif

   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
// Allocate GPU buffers.
void BasePublisherComponent::allocateCudaBuffers() {
   std::size_t const sizeExt = (std::size_t)mActivity->getBufferSizeAcrossBatch() * sizeof(float);
   if (mAllocCudaDatastore) {
      mCudaDatastore = mCudaDevice->createBuffer(sizeExt, &getDescription());
#ifdef PV_USE_CUDNN
      mCudnnDatastore = mCudaDevice->createBuffer(sizeExt, &getDescription());
#endif
   }

   if (mAllocCudaActiveIndices) {
      auto const nbatch = (std::size_t)mActivity->getLayerLoc()->nbatch;
      mCudaNumActive    = mCudaDevice->createBuffer(nbatch * sizeof(long), &getDescription());

      int const numExtendedBatch = mActivity->getBufferSizeAcrossBatch();
      std::size_t const sizeExt  = (std::size_t)numExtendedBatch * sizeof(SparseList<float>::Entry);
      mCudaActiveIndices         = mCudaDevice->createBuffer(sizeExt, &getDescription());
   }
}
#endif // PV_USE_CUDA

Response::Status BasePublisherComponent::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   mPublisher->checkpointDataStore(checkpointer, getName(), "Delays");

   // Timers

   mPublishTimer = new Timer(getName(), "layer", "publish");
   checkpointer->registerTimer(mPublishTimer);

   return Response::SUCCESS;
}

Response::Status BasePublisherComponent::processCheckpointRead() {
   updateAllActiveIndices();
   return Response::SUCCESS;
}

Response::Status BasePublisherComponent::readStateFromCheckpoint(Checkpointer *checkpointer) {
   Response::Status status = BaseObject::readStateFromCheckpoint(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   checkpointer->readNamedCheckpointEntry(std::string(name), std::string("Delays"), false);
   return Response::SUCCESS;
}

Response::Status BasePublisherComponent::respondLayerAdvanceDataStore(
      std::shared_ptr<LayerAdvanceDataStoreMessage const> message) {
   advanceDataStore();
   return Response::SUCCESS;
}

void BasePublisherComponent::advanceDataStore() { mPublisher->increaseTimeLevel(); }

Response::Status
BasePublisherComponent::respondLayerPublish(std::shared_ptr<LayerPublishMessage const> message) {
   mPublishTimer->start();
   publish(mCommunicator, message->mTime);
   mPublishTimer->stop();
   return Response::SUCCESS;
}

void BasePublisherComponent::publish(Communicator const *comm, double simTime) {
   double lastUpdateTime = mUpdateController ? mUpdateController->getLastUpdateTime() : 0.0;
   if (lastUpdateTime >= simTime) {
      if (mBoundaryConditions->getMirrorBCflag()) {
         auto *activityData = mActivity->getReadWritePointer();
         mBoundaryConditions->applyBoundaryConditions(activityData, mActivity->getLayerLoc());
      }
      mPublisher->publish(lastUpdateTime);
#ifdef PV_USE_CUDA
      setUpdatedCudaDatastoreFlag(true);
#endif // PV_USE_CUDA
   }
   else {
      mPublisher->copyForward(lastUpdateTime);
   }
}

// Updates active indices for all levels (delays) here
void BasePublisherComponent::updateAllActiveIndices() { mPublisher->updateAllActiveIndices(); }

void BasePublisherComponent::updateActiveIndices() { mPublisher->updateActiveIndices(0); }

bool BasePublisherComponent::isExchangeFinished(int delay) {
   return mPublisher->isExchangeFinished(delay);
}

int BasePublisherComponent::waitOnPublish(Communicator const *comm) {
   mPublishTimer->start();

   // wait for MPI border transfers to complete
   //
   int status = mPublisher->wait();

   mPublishTimer->stop();
   return status;
}

Response::Status BasePublisherComponent::respondLayerCheckNotANumber(
      std::shared_ptr<LayerCheckNotANumberMessage const> message) {
   Response::Status status = Response::SUCCESS;
   auto layerData          = getLayerData();
   int const N             = mActivity->getBufferSizeAcrossBatch();
   for (int n = 0; n < N; n++) {
      float a = layerData[n];
      FatalIf(
            a != a,
            "%s has not-a-number values in the activity buffer.  Exiting.\n",
            getDescription_c());
   }
   return status;
}

float const *BasePublisherComponent::getLayerData(int delay) const {
   PVLayerCube cube = mPublisher->createCube(delay);
   return cube.data;
}

} // namespace PV
