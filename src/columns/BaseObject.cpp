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

int BaseObject::initialize_base() { return PV_SUCCESS; }

int BaseObject::initialize(const char *name, HyPerCol *hc) {
   setParent(hc);
   return ParamsInterface::initialize(name, hc->parameters());
}

void BaseObject::setParent(HyPerCol *hc) {
   pvAssert(parent == nullptr);
   parent = hc;
}

Response::Status BaseObject::respond(std::shared_ptr<BaseMessage const> message) {
   // TODO: convert PV_SUCCESS, PV_FAILURE, etc. to enum
   Response::Status status = CheckpointerDataInterface::respond(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (message == nullptr) {
      return Response::NO_ACTION;
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
   Response::Status status = Response::NO_ACTION;
   if (getInitInfoCommunicatedFlag()) {
      return status;
   }
   status = communicateInitInfo(message);
   if (Response::completed(status)) {
      setInitInfoCommunicatedFlag();
   }
   return status;
}

#ifdef PV_USE_CUDA
Response::Status
BaseObject::respondSetCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   return setCudaDevice(message);
}
#endif // PV_USE_CUDA

Response::Status
BaseObject::respondAllocateData(std::shared_ptr<AllocateDataMessage const> message) {
   Response::Status status = Response::NO_ACTION;
   if (getDataStructuresAllocatedFlag()) {
      return status;
   }
   status = allocateDataStructures();
   if (Response::completed(status)) {
      setDataStructuresAllocatedFlag();
   }
   return status;
}

Response::Status
BaseObject::respondInitializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = Response::NO_ACTION;
   if (getInitialValuesSetFlag()) {
      return Response::NO_ACTION;
   }
   status = initializeState();
   if (Response::completed(status)) {
      setInitialValuesSetFlag();
   }
   return status;
}

Response::Status BaseObject::respondCopyInitialStateToGPU(
      std::shared_ptr<CopyInitialStateToGPUMessage const> message) {
   return copyInitialStateToGPU();
}

Response::Status BaseObject::respondCleanup(std::shared_ptr<CleanupMessage const> message) {
   return cleanup();
}

#ifdef PV_USE_CUDA
Response::Status BaseObject::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   if (mUsingGPUFlag) {
      mCudaDevice = message->mCudaDevice;
      FatalIf(
            mCudaDevice == nullptr,
            "%s received SetCudaDevice with null device pointer.\n",
            getDescription_c());
   }
   return Response::SUCCESS;
}
#endif // PV_USE_CUDA

BaseObject::~BaseObject() {}

} /* namespace PV */
