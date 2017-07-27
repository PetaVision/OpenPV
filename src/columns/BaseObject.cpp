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
   return parent->parameters()->groupKeywordFromName(getName());
}

int BaseObject::setName(char const *name) {
   pvAssert(this->name == NULL);
   int status = PV_SUCCESS;
   this->name = strdup(name);
   if (this->name == NULL) {
      ErrorLog().printf("could not set name \"%s\": %s\n", name, strerror(errno));
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

int BaseObject::ioParams(enum ParamsIOFlag ioFlag, bool printHeader, bool printFooter) {
   if (printHeader) {
      parent->ioParamsStartGroup(ioFlag, name);
   }
   ioParamsFillGroup(ioFlag);
   if (printFooter) {
      parent->ioParamsFinishGroup(ioFlag);
   }

   return PV_SUCCESS;
}

int BaseObject::respond(std::shared_ptr<BaseMessage const> message) {
   // TODO: convert PV_SUCCESS, PV_FAILURE, etc. to enum
   int status = CheckpointerDataInterface::respond(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   if (message == nullptr) {
      return PV_SUCCESS;
   }
   else if (
         auto castMessage = std::dynamic_pointer_cast<CommunicateInitInfoMessage const>(message)) {
      return respondCommunicateInitInfo(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<AllocateDataMessage const>(message)) {
      return respondAllocateData(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<InitializeStateMessage const>(message)) {
      return respondInitializeState(castMessage);
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<CopyInitialStateToGPUMessage const>(message)) {
      return respondCopyInitialStateToGPU(castMessage);
   }
   else if (auto castMessage = std::dynamic_pointer_cast<CleanupMessage const>(message)) {
      return respondCleanup(castMessage);
   }
   else {
      return PV_SUCCESS;
   }
}

int BaseObject::respondCommunicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = PV_SUCCESS;
   if (getInitInfoCommunicatedFlag()) {
      return status;
   }
   status = communicateInitInfo(message);
   if (status == PV_SUCCESS) {
      setInitInfoCommunicatedFlag();
   }
   return status;
}

int BaseObject::respondAllocateData(std::shared_ptr<AllocateDataMessage const> message) {
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

int BaseObject::respondRegisterData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   int status = registerData(message->mDataRegistry);
   if (status != PV_SUCCESS) {
      Fatal() << getDescription() << ": registerData failed.\n";
   }
   return status;
}

int BaseObject::respondInitializeState(std::shared_ptr<InitializeStateMessage const> message) {
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

int BaseObject::respondCopyInitialStateToGPU(
      std::shared_ptr<CopyInitialStateToGPUMessage const> message) {
   return copyInitialStateToGPU();
}

int BaseObject::respondCleanup(std::shared_ptr<CleanupMessage const> message) { return cleanup(); }

BaseObject::~BaseObject() { free(name); }

} /* namespace PV */
