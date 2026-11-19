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
   delete mInitialPublishTimer;
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
         getName(),
         *mCommunicator->getLocalMPIBlock(),
         mActivity,
         getNumDelayLevels(),
         mSparseLayerFlag,
         mActivity->getBroadcastFlag());
#ifdef PV_USE_CUDA
   allocateCudaBuffers();
#endif

   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
// Allocate GPU buffers.
void BasePublisherComponent::allocateCudaBuffers() {
   std::size_t const sizeExt = (std::size_t)mActivity->getBufferSizeAcrossBatch() * sizeof(float);
   if (mAllocCudaDataStore) {
      mCudaDataStore = mCudaDevice->createBuffer(sizeExt, &getDescription());
#ifdef PV_USE_CUDNN
      mCudnnDataStore = mCudaDevice->createBuffer(sizeExt, &getDescription());
#endif
   }

   if (mAllocCudaActiveIndices) {
      auto const nbatch = (std::size_t)mActivity->getLayerLoc()->nbatch;
      mCudaNumActive    = mCudaDevice->createBuffer(nbatch * sizeof(long), &getDescription());

      long const numExtendedBatch  = mActivity->getBufferSizeAcrossBatch();
      std::size_t const bufferSize =
            (std::size_t)numExtendedBatch * sizeof(SparseEntry<float>);
      mCudaActiveIndices = mCudaDevice->createBuffer(bufferSize, &getDescription());
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
   if (mNumDelayLevels > 1) {
      // If only one delay level, _Delays.pvp file will be the same as _A.pvp file, so we don't
      // waste resources saving it. When reading from checkpoint, we'll need to copy from activity
      mPublisher->checkpointDataStore(checkpointer, getName(), "Delays");
   }

   // Timers

   mInitialPublishTimer = new Timer(getName(), "layer", "initpub");
   checkpointer->registerTimer(mInitialPublishTimer);
   mPublishTimer = new Timer(getName(), "layer", "publish");
   checkpointer->registerTimer(mPublishTimer);

   checkpointer->registerTimer(mPublisher->getIncreaseLevelTimer());
   checkpointer->registerTimer(mPublisher->getIncreaseLevelWaitTimer());

   return Response::SUCCESS;
}

Response::Status BasePublisherComponent::processCheckpointRead(double simTime) {
   if (mNumDelayLevels == 1) {
      // We didn't write a _Delays.pvp file to checkpoint, so we have to copy it from activity.
      mPublisher->publish(simTime);
      mPublisher->wait(0 /*delay*/);
      // Publish::publish() does nonblocking MPI border exchange, so we have to clear the requests.
   }
   updateAllActiveIndices();
   return Response::SUCCESS;
}

Response::Status BasePublisherComponent::readStateFromCheckpoint(Checkpointer *checkpointer) {
   Response::Status status = BaseObject::readStateFromCheckpoint(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   if (mNumDelayLevels == 1) {
      checkpointer->readNamedCheckpointEntry(std::string(mName), std::string("A"), false);
      mPublisher->publish(0.0 /*simTime*/);
      mPublisher->wait(0 /*delay*/);
      // Publish::publish() does nonblocking MPI border exchange, so we have to clear the requests.
   }
   else {
      checkpointer->readNamedCheckpointEntry(std::string(mName), std::string("Delays"), false);
   }
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
   Timer *timer = message->mTime ? mPublishTimer : mInitialPublishTimer;
   timer->start();
   publish(mCommunicator, message->mTime);
   timer->stop();
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
      setUpdatedCudaDataStoreFlag(true);
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

Response::Status BasePublisherComponent::respondLayerCheckNotANumber(
      std::shared_ptr<LayerCheckNotANumberMessage const> message) {
   Response::Status status = Response::SUCCESS;
   auto layerCube          = getLayerCube();
   int const nbatch        = mActivity->getLayerLoc()->nbatch;
   if (layerCube.isSparse) {
      for (int b = 0; b < nbatch; ++b) {
         SparseList<float> const *sparseList = layerCube.activeIndices[b];
         std::vector<SparseEntry<float>> const &sparseData = sparseList->getContents();
         for (auto const &e : sparseData) {
            float a = e.value;
            FatalIf(
                  a != a,
                  "%s has not-a-number values in the activity buffer.  Exiting.\n",
                  getDescription_c());
         }
      }
   }
   else {
      long const N = mActivity->getBufferSize();
      for (int b = 0; b < nbatch; ++b) {
         for (long n = 0L; n < N; ++n) {
            float a = layerCube.data[b][n];
            FatalIf(
                  a != a,
                  "%s has not-a-number values in the activity buffer.  Exiting.\n",
                  getDescription_c());
         }
      }
   }
   return status;
}

Response::Status BasePublisherComponent::cleanup() { return mPublisher->cleanup(); }

} // namespace PV
