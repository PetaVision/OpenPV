/*
 * BaseObject.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include "BaseObject.hpp"
#include "columns/HyPerCol.hpp"
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace PV {

BaseObject::BaseObject() {
   initialize_base();
   // Note that initialize() is not called in the constructor.
   // Instead, derived classes should call BaseObject::initialize in their own
   // constructor.
}

int BaseObject::initialize_base() {
   name   = NULL;
   parent = NULL;
   return PV_SUCCESS;
}

int BaseObject::initialize(const char *name, HyPerCol *hc) {
   int status = setName(name);
   if (status == PV_SUCCESS) {
      status = setParent(hc);
   }
   if (status == PV_SUCCESS) {
      status = setDescription();
   }
   return status;
}

char const *BaseObject::getKeyword() const {
   return getParent()->parameters()->groupKeywordFromName(getName());
}

int BaseObject::setName(char const *name) {
   pvAssert(this->name == NULL);
   int status = PV_SUCCESS;
   this->name = strdup(name);
   if (this->name == NULL) {
      pvErrorNoExit().printf("could not set name \"%s\": %s\n", name, strerror(errno));
      status = PV_FAILURE;
   }
   return status;
}

int BaseObject::setParent(HyPerCol *hc) {
   pvAssert(parent == NULL);
   HyPerCol *parentCol = dynamic_cast<HyPerCol *>(hc);
   int status          = parentCol != NULL ? PV_SUCCESS : PV_FAILURE;
   if (parentCol) {
      parent = parentCol;
   }
   return status;
}

int BaseObject::setDescription() {
   description.clear();
   description.append(getKeyword()).append(" \"").append(getName()).append("\"");
   return PV_SUCCESS;
}

int BaseObject::respond(std::shared_ptr<BaseMessage const> message) {
   // TODO: convert PV_SUCCESS, PV_FAILURE, etc. to enum
   if (message == nullptr) {
      return PV_SUCCESS;
   }
   else if (
         CommunicateInitInfoMessage const *castMessage =
               dynamic_cast<CommunicateInitInfoMessage const *>(message.get())) {
      return respondCommunicateInitInfo(castMessage);
   }
   else if (
         AllocateDataMessage const *castMessage =
               dynamic_cast<AllocateDataMessage const *>(message.get())) {
      return respondAllocateData(castMessage);
   }
   else if (
         RegisterDataMessage<Secretary> const *castMessage =
               dynamic_cast<RegisterDataMessage<Secretary> const *>(message.get())) {
      return respondRegisterData(castMessage);
   }
   else if (
         InitializeStateMessage const *castMessage =
               dynamic_cast<InitializeStateMessage const *>(message.get())) {
      return respondInitializeState(castMessage);
   }
   else {
      return PV_SUCCESS;
   }
}

int BaseObject::respondCommunicateInitInfo(CommunicateInitInfoMessage const *message) {
   int status = PV_SUCCESS;
   if (getInitInfoCommunicatedFlag()) {
      return status;
   }
   status = communicateInitInfo();
   if (status == PV_SUCCESS) {
      setInitInfoCommunicatedFlag();
   }
   return status;
}

int BaseObject::respondAllocateData(AllocateDataMessage const *message) {
   int status = PV_SUCCESS;
   if (getDataStructuresAllocatedFlag()) {
      return status;
   }
   status = allocateDataStructures();
   if (status == PV_SUCCESS) {
      setDataStructuresAllocatedFlag();
   }
   return status;
}

int BaseObject::respondRegisterData(RegisterDataMessage<Secretary> const *message) {
   int status = registerData(message->mDataRegistry, name);
   if (status != PV_SUCCESS) {
      pvError() << getDescription() << ": registerData failed.\n";
   }
   return status;
}

int BaseObject::respondInitializeState(InitializeStateMessage const *message) {
   int status = PV_SUCCESS;
   if (getInitialValuesSetFlag()) {
      return status;
   }
   status = initializeState();
   if (status == PV_SUCCESS) {
      setInitialValuesSetFlag();
   }
   return status;
}

BaseObject::~BaseObject() { free(name); }

} /* namespace PV */
