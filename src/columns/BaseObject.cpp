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
      setDescription();
   }
   return status;
}

char const *BaseObject::lookupKeyword() const {
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

void BaseObject::setDescription() {
   setObjectType();
   description = getObjectType() + " \"" + getName() + "\"";
}

void BaseObject::setObjectType() { mObjectType = lookupKeyword(); }

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

Response::Status BaseObject::respond(std::shared_ptr<BaseMessage const> message) {
   // TODO: convert PV_SUCCESS, PV_FAILURE, etc. to enum
   Response::Status status = CheckpointerDataInterface::respond(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   if (message == nullptr) {
      return Response::SUCCESS;
   }
   else if (
         auto castMessage = std::dynamic_pointer_cast<CommunicateInitInfoMessage const>(message)) {
      return respondCommunicateInitInfo(castMessage);
   }
#ifdef PV_USE_CUDA
   else if (auto castMessage = std::dynamic_pointer_cast<SetCudaDeviceMessage const>(message)) {
      return respondSetCudaDevice(castMessage);
   }
#endif // PV_USE_CUDA
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
      return Response::SUCCESS;
   }
}

Response::Status
BaseObject::respondCommunicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (getInitInfoCommunicatedFlag()) {
      return status;
   }
   int intStatus = communicateInitInfo(message);
   status        = Response::convertIntToStatus(intStatus);
   if (status == Response::SUCCESS) {
      setInitInfoCommunicatedFlag();
   }
   return status;
}

#ifdef PV_USE_CUDA
Response::Status
BaseObject::respondSetCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   int status = setCudaDevice(message);
   return Response::convertIntToStatus(status);
}
#endif // PV_USE_CUDA

Response::Status
BaseObject::respondAllocateData(std::shared_ptr<AllocateDataMessage const> message) {
   Response::Status status = Response::SUCCESS;
   if (getDataStructuresAllocatedFlag()) {
      return status;
   }
   status = Response::convertIntToStatus(allocateDataStructures());
   if (status == Response::SUCCESS) {
      setDataStructuresAllocatedFlag();
   }
   return status;
}

Response::Status
BaseObject::respondRegisterData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   int status = registerData(message->mDataRegistry);
   if (status != PV_SUCCESS) {
      Fatal() << getDescription() << ": registerData failed.\n";
   }
   return Response::SUCCESS;
}

Response::Status
BaseObject::respondInitializeState(std::shared_ptr<InitializeStateMessage const> message) {
   int status = PV_SUCCESS;
   if (getInitialValuesSetFlag()) {
      return Response::SUCCESS;
   }
   status = initializeState();
   if (status == PV_SUCCESS) {
      setInitialValuesSetFlag();
   }
   return Response::convertIntToStatus(status);
}

Response::Status BaseObject::respondCopyInitialStateToGPU(
      std::shared_ptr<CopyInitialStateToGPUMessage const> message) {
   return Response::convertIntToStatus(copyInitialStateToGPU());
}

Response::Status BaseObject::respondCleanup(std::shared_ptr<CleanupMessage const> message) {
   return Response::convertIntToStatus(cleanup());
}

#ifdef PV_USE_CUDA
int BaseObject::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   if (mUsingGPUFlag) {
      mCudaDevice = message->mCudaDevice;
      FatalIf(
            mCudaDevice == nullptr,
            "%s received SetCudaDevice with null device pointer.\n",
            getDescription_c());
   }
   return PV_SUCCESS;
}
#endif // PV_USE_CUDA

BaseObject::~BaseObject() { free(name); }

} /* namespace PV */
