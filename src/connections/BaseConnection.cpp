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
   deleteComponents();
   delete mIOTimer;
}

int BaseConnection::initialize(char const *name, HyPerCol *hc) {
   int status = BaseObject::initialize(name, hc);

   if (status == PV_SUCCESS) {
      defineComponents();
      readParams();
   }
   return status;
}

void BaseConnection::addObserver(Observer *observer) {
   mComponentTable.addObject(observer->getDescription(), observer);
}

void BaseConnection::defineComponents() {
   mConnectionData = createConnectionData();
   if (mConnectionData) {
      addObserver(mConnectionData);
   }
   mDeliveryObject = createDeliveryObject();
   if (mDeliveryObject) {
      addObserver(mDeliveryObject);
   }
}

ConnectionData *BaseConnection::createConnectionData() { return new ConnectionData(name, parent); }

BaseDelivery *BaseConnection::createDeliveryObject() { return new BaseDelivery(name, parent); }

int BaseConnection::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   for (auto &c : mComponentTable.getObjectVector()) {
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
   auto status = notify(
         mComponentTable, message, parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
   return status;
}

Response::Status
BaseConnection::respondConnectionOutput(std::shared_ptr<ConnectionOutputMessage const> message) {
   mIOTimer->start();
   auto status = notify(
         mComponentTable, message, parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
   mIOTimer->stop();
   return status;
}

Response::Status
BaseConnection::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   // build a CommunicateInitInfoMessage consisting of everything in the passed message
   // and everything in the observer table. This way components can communicate with
   // other objects in the HyPerCol's hierarchy.
   auto componentTable = mComponentTable;
   ObjectMapComponent objectMapComponent(name, parent);
   objectMapComponent.setObjectMap(message->mHierarchy);
   componentTable.addObject(objectMapComponent.getDescription(), &objectMapComponent);
   auto communicateMessage =
         std::make_shared<CommunicateInitInfoMessage>(componentTable.getObjectMap());

   Response::Status status = notify(
         componentTable,
         communicateMessage,
         parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);

   if (Response::completed(status)) {
      auto *deliveryObject = getComponentByType<BaseDelivery>();
      pvAssert(deliveryObject);
      HyPerLayer *postLayer = deliveryObject->getPostLayer();
      if (postLayer != nullptr) {
         postLayer->addRecvConn(this);
      }
#ifdef PV_USE_CUDA
      for (auto &c : componentTable.getObjectVector()) {
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
   status = notify(
         mComponentTable, message, parent->getCommunicator()->globalCommunicator() /*printFlag*/);
   return status;
}
#endif // PV_USE_CUDA

Response::Status BaseConnection::allocateDataStructures() {
   Response::Status status = notify(
         mComponentTable,
         std::make_shared<AllocateDataMessage>(),
         parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
   return status;
}

Response::Status BaseConnection::registerData(Checkpointer *checkpointer) {
   auto status = notify(
         mComponentTable,
         std::make_shared<RegisterDataMessage<Checkpointer>>(checkpointer),
         parent->getCommunicator()->globalCommRank() == 0 /*printFlag*/);
   mIOTimer = new Timer(getName(), "conn", "io");
   checkpointer->registerTimer(mIOTimer);
   return status;
}

void BaseConnection::deleteComponents() {
   mComponentTable.clear(true); // Deletes each component and clears the component table
}

} // namespace PV
