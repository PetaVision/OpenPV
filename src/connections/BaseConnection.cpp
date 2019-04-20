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

BaseConnection::~BaseConnection() { delete mIOTimer; }

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

void BaseConnection::fillComponentTable() {
   ComponentBasedObject::fillComponentTable();
   auto *connectionData = createConnectionData();
   if (connectionData) {
      addUniqueComponent(connectionData);
   }
   auto *deliveryObject = createDeliveryObject();
   if (deliveryObject) {
      addUniqueComponent(deliveryObject);
   }
}

ConnectionData *BaseConnection::createConnectionData() {
   return new ConnectionData(name, parameters(), mCommunicator);
}

BaseDelivery *BaseConnection::createDeliveryObject() {
   return new BaseDelivery(name, parameters(), mCommunicator);
}

int BaseConnection::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ComponentBasedObject::ioParamsFillGroup(ioFlag);
   return status;
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
   Response::Status status = ComponentBasedObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

#ifdef PV_USE_CUDA
   for (auto &c : *mTable) {
      auto *baseObject = dynamic_cast<BaseObject *>(c);
      if (baseObject) {
         mUsingGPUFlag |= baseObject->isUsingGPU();
      }
   }
#endif // PV_USE_CUDA
   status = Response::SUCCESS;

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

Response::Status
BaseConnection::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ComponentBasedObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }

   mIOTimer = new Timer(getName(), "conn", "io");
   message->mDataRegistry->registerTimer(mIOTimer);
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
