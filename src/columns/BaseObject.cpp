/*
 * BaseObject.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include "BaseObject.hpp"
#include "columns/Communicator.hpp"
#include "columns/Factory.hpp"
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace PV {

BaseObject::BaseObject() {
   // Note that initialize() is not called in the constructor.
   // Instead, derived classes should call BaseObject::initialize in their own
   // constructor.
}

void BaseObject::initialize(const char *name, PVParams *params, Communicator const *comm) {
   setCommunicator(comm);
   ParamsInterface::initialize(name, params);
}

void BaseObject::setCommunicator(Communicator const *comm) {
   pvAssert(mCommunicator == nullptr);
   mCommunicator = comm;
}

void BaseObject::initMessageActionMap() {
   ParamsInterface::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<CommunicateInitInfoMessage const>(msgptr);
      return respondCommunicateInitInfo(castMessage);
   };
   mMessageActionMap.emplace("CommunicateInitInfo", action);

#ifdef PV_USE_CUDA
   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<SetCudaDeviceMessage const>(msgptr);
      return respondSetCudaDevice(castMessage);
   };
   mMessageActionMap.emplace("SetCudaDevice", action);
#endif // PV_USE_CUDA

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<AllocateDataStructuresMessage const>(msgptr);
      return respondAllocateDataStructures(castMessage);
   };
   mMessageActionMap.emplace("AllocateDataStructures", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<InitializeStateMessage const>(msgptr);
      return respondInitializeState(castMessage);
   };
   mMessageActionMap.emplace("InitializeState", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<CopyInitialStateToGPUMessage const>(msgptr);
      return respondCopyInitialStateToGPU(castMessage);
   };
   mMessageActionMap.emplace("CopyInitialStateToGPU", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<CleanupMessage const>(msgptr);
      return respondCleanup(castMessage);
   };
   mMessageActionMap.emplace("Cleanup", action);
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

Response::Status BaseObject::respondAllocateDataStructures(
      std::shared_ptr<AllocateDataStructuresMessage const> message) {
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
   status = initializeState(message);
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
   mCudaDevice = message->mCudaDevice;
   FatalIf(
         mCudaDevice == nullptr,
         "%s received SetCudaDevice with null device pointer.\n",
         getDescription_c());
   return Response::SUCCESS;
}
#endif // PV_USE_CUDA

BaseObject *BaseObject::createSubobject(char const *keyword) {
   BaseObject *subobject = nullptr;
   try {
      subobject = Factory::instance()->createByKeyword(keyword, name, parameters(), mCommunicator);
   } catch (const std::exception &e) {
      Fatal() << getDescription() << " unable to create " << keyword << ": " << e.what() << "\n";
   }
   FatalIf(
         subobject == nullptr, // Because of try/catch above, this should never happen.
         "%s attempt to create %s returned null pointer.\n",
         getDescription_c(),
         keyword);
   return subobject;
}

BaseObject::~BaseObject() {}

} /* namespace PV */
