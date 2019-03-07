/*
 * BaseConnection.cpp
 *
 *  Created on Sep 19, 2014
 *      Author: Pete Schultz
 */

#include "BaseConnection.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

BaseConnection::BaseConnection(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

BaseConnection::BaseConnection() {}

BaseConnection::~BaseConnection() {}

void BaseConnection::initialize(char const *name, PVParams *params, Communicator const *comm) {
   ComponentBasedObject::initialize(name, params, comm);

   // The WeightsPair writes this flag to output params file. Other ParamsInterface-derived
   // components of the connection will automatically read InitializeFromCheckpointFlag, but
   // shouldn't also write it.
   mWriteInitializeFromCheckpointFlag = true;
}

void BaseConnection::initMessageActionMap() {
   ComponentBasedObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionWriteParamsMessage const>(msgptr);
      return respondConnectionWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ConnectionWriteParams", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionFinalizeUpdateMessage const>(msgptr);
      return respondConnectionFinalizeUpdate(castMessage);
   };
   mMessageActionMap.emplace("ConnectionFinalizeUpdate", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ConnectionOutputMessage const>(msgptr);
      return respondConnectionOutput(castMessage);
   };
   mMessageActionMap.emplace("ConnectionOutput", action);
}

void BaseConnection::createComponentTable(char const *description) {
   ComponentBasedObject::createComponentTable(description);
   auto *connectionData = createConnectionData();
   if (connectionData) {
      addUniqueComponent(connectionData->getDescription(), connectionData);
   }
   auto *deliveryObject = createDeliveryObject();
   if (deliveryObject) {
      addUniqueComponent(deliveryObject->getDescription(), deliveryObject);
   }
}

ConnectionData *BaseConnection::createConnectionData() {
   return new ConnectionData(name, parameters(), mCommunicator);
}

BaseDelivery *BaseConnection::createDeliveryObject() {
   return new BaseDelivery(name, parameters(), mCommunicator);
}

int BaseConnection::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   for (auto &c : *mTable) {
      auto obj = dynamic_cast<BaseObject *>(c);
      if (obj) {
         obj->ioParams(ioFlag, false, false);
      }
   }
   return PV_SUCCESS;
}

Response::Status BaseConnection::respondConnectionWriteParams(
      std::shared_ptr<ConnectionWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

Response::Status BaseConnection::respondConnectionFinalizeUpdate(
      std::shared_ptr<ConnectionFinalizeUpdateMessage const> message) {
   auto status = notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   return status;
}

Response::Status
BaseConnection::respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message) {
   mIOTimer->start();
   auto status = notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   mIOTimer->stop();
   return status;
}

Response::Status
BaseConnection::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   // Add a component consisting of a lookup table from the names of HyPerCol's components
   // to the HyPerCol's components themselves. This is needed by, for example, CloneWeightsPair,
   // to find the original weights pair.
   // Since communicateInitInfo can be called more than once, we must ensure that the
   // ObserverTable is only added once.
   auto *tableComponent = mTable->lookupByType<ObserverTable>();
   if (!tableComponent) {
      std::string tableDescription = std::string("ObserverTable \"") + getName() + "\"";
      tableComponent               = new ObserverTable(tableDescription.c_str());
      tableComponent->copyTable(message->mHierarchy);
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

   if (Response::completed(status)) {
#ifdef PV_USE_CUDA
      for (auto &c : *mTable) {
         auto *baseObject = dynamic_cast<BaseObject *>(c);
         if (baseObject) {
            mUsingGPUFlag |= baseObject->isUsingGPU();
         }
      }
#endif // PV_USE_CUDA
      status = Response::SUCCESS;
   }

   return status;
}

#ifdef PV_USE_CUDA
Response::Status
BaseConnection::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   auto status = ComponentBasedObject::setCudaDevice(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   status = notify(message, mCommunicator->globalCommunicator() /*printFlag*/);
   return status;
}
#endif // PV_USE_CUDA

Response::Status BaseConnection::allocateDataStructures() {
   Response::Status status = notify(
         std::make_shared<AllocateDataStructuresMessage>(),
         mCommunicator->globalCommRank() == 0 /*printFlag*/);
   return status;
}

Response::Status
BaseConnection::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto *checkpointer = message->mDataRegistry;
   auto status        = notify(
         std::make_shared<RegisterDataMessage<Checkpointer>>(checkpointer),
         mCommunicator->globalCommRank() == 0 /*printFlag*/);
   mIOTimer = new Timer(getName(), "conn", "io");
   checkpointer->registerTimer(mIOTimer);
   return status;
}

Response::Status
BaseConnection::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
}

Response::Status BaseConnection::copyInitialStateToGPU() {
   auto status = ComponentBasedObject::copyInitialStateToGPU();
   if (status != Response::SUCCESS) {
      return status;
   }
   auto message = std::make_shared<CopyInitialStateToGPUMessage>();
   status       = notify(message, mCommunicator->globalCommunicator() /*printFlag*/);
   return status;
}

} // namespace PV
