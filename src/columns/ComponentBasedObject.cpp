/*
 * ComponentBasedObject.cpp
 *
 *  Created on: Jun 11, 2016
 *      Author: pschultz
 */

#include "ComponentBasedObject.hpp"

namespace PV {

ComponentBasedObject::ComponentBasedObject() {
   // Derived classes should call ComponentBasedObject::initialize() during their own
   // instantiation.
}

void ComponentBasedObject::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   BaseObject::initialize(paramsIO, comm);
   std::string componentTableName = std::string("ObserverTable \"") + getName() + "\"";
   Subject::initializeTable(componentTableName.c_str());
}

int ComponentBasedObject::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   // Components, like all BaseObject-derived objects, read their params during instantiation.
   // When writing out the params file, ComponentBasedObjects must pass the write message
   // to their components.
   if (ioSwitch == ParamsIOSwitch::Write) {
      for (auto *c : *mTable) {
         auto obj = dynamic_cast<BaseObject *>(c);
         if (obj) {
            obj->getParamsIO()->setPrintParamsStream(this->getParamsIO()->getPrintParamsStream());
            obj->getParamsIO()->setPrintLuaStream(this->getParamsIO()->getPrintLuaStream());
            obj->ioParams(ioSwitch, false, false);
         }
      }
   }
   return PV_SUCCESS;
}

Response::Status ComponentBasedObject::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   // Add a component consisting of a lookup table from the names of the parent's components
   // to the components themselves. This is needed by, for example, CloneWeightsPair, to find the
   // original weights pair.
   // Since communicateInitInfo can be called more than once, we must ensure that the
   // ObserverTable is only added once.
   Response::Status status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto communicateMessage = std::make_shared<CommunicateInitInfoMessage>(
         message->mObjectTable,
         message->mDeltaTime,
         message->mNxGlobal,
         message->mNyGlobal,
         message->mNBatchGlobal,
         message->mNumThreads);

   status = status + notify(communicateMessage, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   return status;
}

Response::Status ComponentBasedObject::allocateDataStructures() {
   Response::Status status = BaseObject::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   // Pass on the allocate message to the components.
   auto allocateMessage = std::make_shared<AllocateDataStructuresMessage>();
   status = notify(allocateMessage, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   return status;
}

Response::Status ComponentBasedObject::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   Response::Status status = BaseObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   status = notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   return status;
}

ComponentBasedObject::~ComponentBasedObject() {}

} /* namespace PV */
