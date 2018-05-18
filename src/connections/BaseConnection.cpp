/*
 * BaseConnection.cpp
 *
 *  Created on Sep 19, 2014
 *      Author: Pete Schultz
 */

#include "BaseConnection.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

BaseConnection::BaseConnection(char const *name, HyPerCol *hc) { initialize(name, hc); }

BaseConnection::BaseConnection() {}

BaseConnection::~BaseConnection() {
   delete mIOTimer;
   mObserverTable.clear(true); // deletes the components.
}

int BaseConnection::initialize(char const *name, HyPerCol *hc) {
   int status = BaseObject::initialize(name, hc);

   if (status == PV_SUCCESS) {
      setObserverTable();
      readParams();
   }
   return status;
}

void BaseConnection::initMessageActionMap() {
   ParamsInterface::initMessageActionMap();
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

void BaseConnection::setObserverTable() {
   mConnectionData = createConnectionData();
   if (mConnectionData) {
      addUniqueComponent(mConnectionData->getDescription(), mConnectionData);
   }
   mDeliveryObject = createDeliveryObject();
   if (mDeliveryObject) {
      addUniqueComponent(mDeliveryObject->getDescription(), mDeliveryObject);
   }
}

ConnectionData *BaseConnection::createConnectionData() { return new ConnectionData(name, parent); }

BaseDelivery *BaseConnection::createDeliveryObject() { return new BaseDelivery(name, parent); }

int BaseConnection::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   for (auto &c : mObserverTable) {
      auto obj = dynamic_cast<BaseObject *>(c);
      obj->ioParams(ioFlag, false, false);
   }
   return PV_SUCCESS;
}

Response::Status BaseConnection::respond(std::shared_ptr<BaseMessage const> message) {
   Response::Status status = BaseObject::respond(message);
   if (!Response::completed(status)) {
      return status;
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<ConnectionWriteParamsMessage const>(message)) {
      return respondConnectionWriteParams(castMessage);
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<ConnectionFinalizeUpdateMessage const>(message)) {
      return respondConnectionFinalizeUpdate(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<ConnectionOutputMessage const>(message)) {
      return respondConnectionOutput(castMessage);
   }
   else {
      return status;
   }
}

Response::Status BaseConnection::respondConnectionWriteParams(
      std::shared_ptr<ConnectionWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

Response::Status BaseConnection::respondConnectionFinalizeUpdate(
      std::shared_ptr<ConnectionFinalizeUpdateMessage const> message) {
   auto status = notify(message, parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
   return status;
}

Response::Status
BaseConnection::respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message) {
   mIOTimer->start();
   auto status = notify(message, parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
   mIOTimer->stop();
   return status;
}

Response::Status
BaseConnection::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   // Add a component consisting of a lookup table from the names of HyPerCol's components
   // to the HyPerCol's components themselves. This is needed by, for example, CloneWeightsPair,
   // to find the original weights pair.
   // Since communicateInitInfo can be called more than once, we must ensure that the
   // ObjectMapComponent is only added once.
   auto *objectMapComponent = mapLookupByType<ObjectMapComponent>(mObserverTable.getObjectMap());
   if (!objectMapComponent) {
      objectMapComponent = new ObjectMapComponent(name, parent);
      objectMapComponent->setObjectMap(message->mHierarchy);
      addUniqueComponent(objectMapComponent->getDescription(), objectMapComponent);
      // ObserverTable takes ownership; objectMapComponent will be deleted by
      // Subject::deleteObserverTable() method during destructor.
   }
   pvAssert(objectMapComponent);

   auto communicateMessage =
         std::make_shared<CommunicateInitInfoMessage>(mObserverTable.getObjectMap());

   Response::Status status =
         notify(communicateMessage, parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);

   if (Response::completed(status)) {
      auto *deliveryObject = getComponentByType<BaseDelivery>();
      pvAssert(deliveryObject);
      HyPerLayer *postLayer = deliveryObject->getPostLayer();
      if (postLayer != nullptr) {
         postLayer->addRecvConn(this);
      }
#ifdef PV_USE_CUDA
      for (auto &c : mObserverTable) {
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
   auto status = BaseObject::setCudaDevice(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   status = notify(message, parent->getCommunicator()->globalCommunicator() /*printFlag*/);
   return status;
}
#endif // PV_USE_CUDA

Response::Status BaseConnection::allocateDataStructures() {
   Response::Status status = notify(
         std::make_shared<AllocateDataMessage>(),
         parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
   return status;
}

Response::Status
BaseConnection::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto *checkpointer = message->mDataRegistry;
   auto status        = notify(
         std::make_shared<RegisterDataMessage<Checkpointer>>(checkpointer),
         parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
   mIOTimer = new Timer(getName(), "conn", "io");
   checkpointer->registerTimer(mIOTimer);
   return status;
}

} // namespace PV
