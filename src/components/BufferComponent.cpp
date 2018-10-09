/*
 * BufferComponent.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "BufferComponent.hpp"
#include "checkpointing/CheckpointEntryPvpBuffer.hpp"
#include "columns/HyPerCol.hpp"
#include "components/InitializeFromCheckpointFlag.hpp"

namespace PV {

BufferComponent::BufferComponent(char const *name, HyPerCol *hc) { initialize(name, hc); }

BufferComponent::~BufferComponent() {
#ifdef PV_USE_CUDA
   delete mCudaBuffer;
#endif // PV_USE_CUDA
}

int BufferComponent::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void BufferComponent::setObjectType() { mObjectType = "BufferComponent"; }

Response::Status
BufferComponent::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto hierarchy = message->mHierarchy;
   mLayerGeometry = hierarchy->lookupByType<LayerGeometry>();
   FatalIf(!mLayerGeometry, "%s requires a LayerGeometry component.\n", getDescription_c());
   auto *initializeFromCheckpointComponent =
         hierarchy->lookupByType<InitializeFromCheckpointFlag>();
   mInitializeFromCheckpointFlag =
         initializeFromCheckpointComponent->getInitializeFromCheckpointFlag();
   return Response::SUCCESS;
}

Response::Status BufferComponent::allocateDataStructures() {
   Response::Status status = BaseObject::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   PVLayerLoc const *loc = getLayerLoc();
   int nx                = loc->nx + getExtendedFlag() * (loc->halo.lt + loc->halo.rt);
   int ny                = loc->ny + getExtendedFlag() * (loc->halo.dn + loc->halo.up);
   int nf                = loc->nf;
   int nb                = loc->nbatch;
   mBufferSize           = nx * ny * nf;
   int numNeurons        = mBufferSize * nb * mNumChannels;
   mBufferData.resize(numNeurons);

#ifdef PV_USE_CUDA
   if (mUsingGPUFlag) {
      FatalIf(
            mCudaDevice == nullptr,
            "%s did not receive a SetCudaDevice message.\n",
            getDescription_c());
      std::size_t sizeInBytes = sizeof(float) * (std::size_t)getBufferSizeAcrossBatch();
      mCudaBuffer             = mCudaDevice->createBuffer(sizeInBytes, &getDescription());
   }
// Should the BufferComponent set mUsingGPUFlag in response to SetCudaDevice,
// and eliminate the useCuda() function member?
#endif // PV_USE_CUDA
   return Response::SUCCESS;
}

Response::Status
BufferComponent::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (!mBufferLabel.empty()) {
      auto *checkpointer   = message->mDataRegistry;
      auto checkpointEntry = std::make_shared<CheckpointEntryPvpBuffer<float>>(
            getName(),
            mBufferLabel.c_str(),
            getMPIBlock(),
            mBufferData.data(),
            getLayerLoc(),
            getExtendedFlag());
      bool registerSucceeded =
            checkpointer->registerCheckpointEntry(checkpointEntry, false /*not constant*/);
      FatalIf(
            !registerSucceeded,
            "%s failed to register %s for checkpointing.\n",
            getDescription_c(),
            getBufferData());
   }
   return Response::SUCCESS;
}

Response::Status BufferComponent::readStateFromCheckpoint(Checkpointer *checkpointer) {
   Response::Status status = BaseObject::readStateFromCheckpoint(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   if (!mBufferLabel.empty() and mInitializeFromCheckpointFlag) {
      checkpointer->readNamedCheckpointEntry(std::string(name), mBufferLabel, false);
   }
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
void BufferComponent::useCuda() {
   FatalIf(
         getDataStructuresAllocatedFlag(),
         "%s cannot set the UsingGPU flag after data allocation.\n",
         getDescription_c());
   mUsingGPUFlag = true;
}

void BufferComponent::copyFromCuda() {
   FatalIf(!mCudaBuffer, "%s did not allocate its CudaBuffer.\n", getDescription_c());
   mCudaBuffer->copyFromDevice(mBufferData.data());
}

void BufferComponent::copyToCuda() {
   FatalIf(!mCudaBuffer, "%s did not allocate its CudaBuffer.\n", getDescription_c());
   mCudaBuffer->copyToDevice(mBufferData.data());
}
#endif // PV_USE_CUDA

} // namespace PV
