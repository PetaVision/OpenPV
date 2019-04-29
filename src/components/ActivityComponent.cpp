/*
 * ActivityComponent.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "ActivityComponent.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

ActivityComponent::ActivityComponent(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ActivityComponent::~ActivityComponent() {
   delete mUpdateTimer;
#ifdef PV_USE_CUDA
   delete mUpdateCudaTimer;
#endif // PV_USE_CUDA
}

void ActivityComponent::initialize(char const *name, PVParams *params, Communicator const *comm) {
   ComponentBasedObject::initialize(name, params, comm);
}

void ActivityComponent::setObjectType() { mObjectType = "ActivityComponent"; }

int ActivityComponent::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ComponentBasedObject::ioParamsFillGroup(ioFlag);

   // GPU-specific parameter.  If not using GPUs, this flag can be set to false or left out,
   // but it is an error to set updateGpu to true if compiling without GPUs.  We read it here and
   // not in any component because it will typically need to be broadcast to several components
   // (which happens during the communicate stage).
   ioParam_updateGpu(ioFlag);

   return PV_SUCCESS;
}

void ActivityComponent::ioParam_updateGpu(enum ParamsIOFlag ioFlag) {
#ifdef PV_USE_CUDA
   parameters()->ioParamValue(
         ioFlag, name, "updateGpu", &mUpdateGpu, mUpdateGpu, true /*warnIfAbsent*/);
#else // PV_USE_CUDA
   bool mUpdateGpu = false;
   parameters()->ioParamValue(
         ioFlag, name, "updateGpu", &mUpdateGpu, mUpdateGpu, false /*warnIfAbsent*/);
   if (mCommunicator->globalCommRank() == 0) {
      FatalIf(
            mUpdateGpu,
            "%s: updateGpu is set to true, but PetaVision was compiled without GPU acceleration.\n",
            getDescription_c());
   }
#endif // PV_USE_CUDA
}

void ActivityComponent::fillComponentTable() {
   ComponentBasedObject::fillComponentTable();
   mActivity = createActivity();
   if (mActivity) {
      addUniqueComponent(mActivity);
   }
}

ActivityBuffer *ActivityComponent::createActivity() {
   return new ActivityBuffer(getName(), parameters(), mCommunicator);
}

Response::Status
ActivityComponent::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = ComponentBasedObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

#ifdef PV_USE_CUDA
   if (mUpdateGpu) {
      useCuda();
   }
#endif // PV_USE_CUDA

   return Response::SUCCESS;
}

Response::Status
ActivityComponent::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ComponentBasedObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }

   // Timers

   auto *checkpointer = message->mDataRegistry;
   mUpdateTimer       = new Timer(getName(), "layer", "update ");
   checkpointer->registerTimer(mUpdateTimer);

#ifdef PV_USE_CUDA
   auto cudaDevice = mCudaDevice;
   if (cudaDevice) {
      mUpdateCudaTimer = new PVCuda::CudaTimer(getName(), "layer", "gpuupdate");
      mUpdateCudaTimer->setStream(cudaDevice->getStream());
      checkpointer->registerTimer(mUpdateCudaTimer);
   }
#endif // PV_USE_CUDA

   return Response::SUCCESS;
}

Response::Status
ActivityComponent::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return mActivity->respond(message);
}

Response::Status ActivityComponent::readStateFromCheckpoint(Checkpointer *checkpointer) {
   Response::Status status = ComponentBasedObject::readStateFromCheckpoint(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   auto message = std::make_shared<ReadStateFromCheckpointMessage<Checkpointer>>(checkpointer);
   return notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
}

#ifdef PV_USE_CUDA
Response::Status
ActivityComponent::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   Response::Status status = ComponentBasedObject::setCudaDevice(message);
   if (Response::completed(status)) {
      status = notify(message, mCommunicator->globalCommRank() == 0 /*printFlag*/);
   }
   return status;
}

Response::Status ActivityComponent::copyInitialStateToGPU() {
   if (mUpdateGpu) {
      Response::Status status = Response::SUCCESS;
      for (auto *obj : *mTable) {
         auto *buffer = dynamic_cast<ComponentBuffer *>(obj);
         if (buffer) {
            pvAssert(buffer->isUsingGPU());
            status = buffer->respond(std::make_shared<CopyInitialStateToGPUMessage>());
            FatalIf(
                  !Response::completed(status),
                  "%s failed to copy initial state to GPU.\n",
                  buffer->getDescription_c());
         }
      }
      return Response::SUCCESS;
   }
   else {
      return Response::NO_ACTION;
   }
}
#endif // PV_USE_CUDA

Response::Status ActivityComponent::updateState(double simTime, double deltaTime) {
   // Move update timers and gpu update timers here.
   mUpdateTimer->start();
#ifdef PV_USE_CUDA
   if (getUpdateGpu()) {
      mUpdateCudaTimer->start();
   }
#endif // PV_USE_CUDA
   auto status = updateActivity(simTime, deltaTime);
#ifdef PV_USE_CUDA
   if (getUpdateGpu()) {
      mUpdateCudaTimer->stop();
   }
#endif // PV_USE_CUDA
   mUpdateTimer->stop();
   return status;
}

Response::Status ActivityComponent::updateActivity(double simTime, double deltaTime) {
   mActivity->updateBuffer(simTime, deltaTime);
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
void ActivityComponent::useCuda() {
   mUsingGPUFlag = mUpdateGpu;
   for (auto *obj : *mTable) {
      auto *buffer = dynamic_cast<ComponentBuffer *>(obj);
      if (buffer) {
         buffer->useCuda();
      }
   }
}

void ActivityComponent::copyToCuda() {
   for (auto *obj : *mTable) {
      auto *buffer = dynamic_cast<ComponentBuffer *>(obj);
      if (buffer) {
         buffer->copyToCuda();
      }
   }
}

void ActivityComponent::copyFromCuda() {
   for (auto *obj : *mTable) {
      auto *buffer = dynamic_cast<ComponentBuffer *>(obj);
      if (buffer && !buffer->getBufferLabel().empty()) {
         buffer->copyFromCuda();
      }
   }
   if (getUpdateGpu()) {
      mUpdateCudaTimer->accumulateTime();
   }
}
#endif // PV_USE_CUDA

} // namespace PV
