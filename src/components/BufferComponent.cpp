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

BufferComponent::~BufferComponent() {}

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
   mLayerGeometry = mapLookupByType<LayerGeometry>(hierarchy);
   FatalIf(!mLayerGeometry, "%s requires a LayerGeometry component.\n", getDescription_c());
   auto *initializeFromCheckpointComponent =
         mapLookupByType<InitializeFromCheckpointFlag>(hierarchy);
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
   int numNeurons        = mBufferSize * nb;
   mBufferData.resize(numNeurons);
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

} // namespace PV
