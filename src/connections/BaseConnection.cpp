/*
 * BaseConnection.cpp
 *
 *  Created on Sep 19, 2014
 *      Author: Pete Schultz
 */

#include "BaseConnection.hpp"
#include "columns/HyPerCol.hpp"
#include "columns/ObjectMapComponent.hpp"

namespace PV {

BaseConnection::BaseConnection(char const *name, HyPerCol *hc) { initialize(name, hc); }

BaseConnection::BaseConnection() {}

BaseConnection::~BaseConnection() { deleteComponents(); }

int BaseConnection::initialize(char const *name, HyPerCol *hc) {
   int status = BaseObject::initialize(name, hc);
   defineComponents();

   if (status == PV_SUCCESS)
      status = readParams();
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
   // mWeightUpdater = createWeightUpdater();
   // if (mWeightUpdater) {
   //    addObserver(mWeightUpdater);
   // }
}

ConnectionData *BaseConnection::createConnectionData() { return new ConnectionData(name, parent); }

// NormalizeBase *BaseConnection::createWeightNormalizer() {
//    NormalizeBase *normalizer = nullptr;
//    parent->parameters()->ioParamString(
//          PARAMS_IO_READ, name, "normalizeMethod", &mNormalizeMethod, NULL, true
//          /*warnIfAbsent*/);
//    if (mNormalizeMethod == nullptr) {
//       if (parent->columnId() == 0) {
//          Fatal().printf(
//                "%s: specifying a normalizeMethod string is required.\n", getDescription_c());
//       }
//    }
//    if (!strcmp(mNormalizeMethod, "")) {
//       free(mNormalizeMethod);
//       mNormalizeMethod = strdup("none");
//    }
//    if (strcmp(mNormalizeMethod, "none")) {
//       BaseObject *baseObj = Factory::instance()->createByKeyword(mNormalizeMethod, name, parent);
//       if (baseObj == nullptr) {
//          if (parent->columnId() == 0) {
//             Fatal() << getDescription_c() << ": normalizeMethod \"" << mNormalizeMethod
//                     << "\" is not recognized." << std::endl;
//          }
//          MPI_Barrier(parent->getCommunicator()->communicator());
//          exit(EXIT_FAILURE);
//       }
//       normalizer = dynamic_cast<NormalizeBase *>(baseObj);
//       if (normalizer == nullptr) {
//          pvAssert(baseObj);
//          if (parent->columnId() == 0) {
//             Fatal() << getDescription_c() << ": normalizeMethod \"" << mNormalizeMethod
//                     << "\" is not a recognized normalization method." << std::endl;
//          }
//          MPI_Barrier(parent->getCommunicator()->communicator());
//          exit(EXIT_FAILURE);
//       }
//    }
//    return normalizer;
// }

BaseDelivery *BaseConnection::createDeliveryObject() { return new BaseDelivery(name, parent); }
//
// WeightUpdater *BaseConnection::createWeightUpdater() {
//    return new WeightUpdater(name, parent);
// }

int BaseConnection::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_initializeFromCheckpointFlag(ioFlag);
   for (auto &c : mComponentTable.getObjectVector()) {
      auto obj = dynamic_cast<BaseObject *>(c);
      obj->ioParams(ioFlag, false, false);
   }
   return PV_SUCCESS;
}

void BaseConnection::ioParam_initializeFromCheckpointFlag(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag,
         name,
         "initializeFromCheckpointFlag",
         &initializeFromCheckpointFlag,
         initializeFromCheckpointFlag,
         true /*warnIfAbsent*/);
}

int BaseConnection::respond(std::shared_ptr<BaseMessage const> message) {
   int status = BaseObject::respond(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<ConnectionWriteParamsMessage const>(message)) {
      return respondConnectionWriteParams(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<ConnectionUpdateMessage const>(message)) {
      return respondConnectionUpdate(castMessage);
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

int BaseConnection::respondConnectionWriteParams(
      std::shared_ptr<ConnectionWriteParamsMessage const> message) {
   return PV_SUCCESS; // return writeParams();
}

int BaseConnection::respondConnectionUpdate(
      std::shared_ptr<ConnectionUpdateMessage const> message) {
   return PV_SUCCESS; // return updateState(message->mTime, message->mDeltaT);
}

int BaseConnection::respondConnectionFinalizeUpdate(
      std::shared_ptr<ConnectionFinalizeUpdateMessage const> message) {
   return PV_SUCCESS; // return finalizeUpdate(message->mTime, message->mDeltaT);
}

int BaseConnection::respondConnectionOutput(
      std::shared_ptr<ConnectionOutputMessage const> message) {
   return PV_SUCCESS; // return outputState(message->mTime);
}

int BaseConnection::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   // build a CommunicateInitInfoMessage consisting of everything in the passed message
   // and everything in the observer table. This way components can communicate with
   // other objects in the HyPerCol's hierarchy.
   auto componentTable = mComponentTable;
   ObjectMapComponent objectMapComponent(name, parent);
   objectMapComponent.setObjectMap(message->mHierarchy);
   componentTable.addObject(objectMapComponent.getDescription(), &objectMapComponent);
   auto communicateMessage =
         std::make_shared<CommunicateInitInfoMessage>(componentTable.getObjectMap());

   notify(
         componentTable,
         communicateMessage,
         parent->getCommunicator()->commRank() == 0 /*printFlag*/);
   return PV_SUCCESS;
}

int BaseConnection::allocateDataStructures() {
   notify(
         mComponentTable,
         std::make_shared<AllocateDataMessage>(),
         parent->getCommunicator()->commRank() == 0 /*printFlag*/);
   return PV_SUCCESS;
}

void BaseConnection::deleteComponents() {
   mComponentTable.clear(true); // Deletes each component and clears the component table
}

} // namespace PV
