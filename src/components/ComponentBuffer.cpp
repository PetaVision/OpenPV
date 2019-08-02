/*
 * ComponentBuffer.cpp
 *
 *  Created on: Oct 12, 2018
 *      Author: Pete Schultz
 */

#include "ComponentBuffer.hpp"
#include "checkpointing/CheckpointEntryPvpBuffer.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

ComponentBuffer::ComponentBuffer(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

ComponentBuffer::~ComponentBuffer() {
#ifdef PV_USE_CUDA
   delete mCudaBuffer;
#endif // PV_USE_CUDA
}

void ComponentBuffer::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
   mBufferLabel = "";
}

void ComponentBuffer::setObjectType() { mObjectType = "ComponentBuffer"; }

void ComponentBuffer::setBufferLabel(std::string const &label) {
   FatalIf(
         !mBufferLabel.empty(),
         "%s called with setBufferLabel(\"%s\"), but the buffer label has already been set.\n",
         getDescription_c(),
         label.c_str());
   mBufferLabel = label;
}

Response::Status
ComponentBuffer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto *objectTable = message->mObjectTable;
   if (mLayerGeometry == nullptr) {
      mLayerGeometry = objectTable->findObject<LayerGeometry>(getName());
   }
   FatalIf(!mLayerGeometry, "%s requires a LayerGeometry component.\n", getDescription_c());
   return Response::SUCCESS;
}

Response::Status ComponentBuffer::allocateDataStructures() {
   Response::Status status = BaseObject::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   setBufferSize();
   setReadOnlyPointer();
   setReadWritePointer();
#ifdef PV_USE_CUDA
   if (mUsingGPUFlag) {
      setCudaBuffer();
      allocateUpdateKernel();
   }
#endif // PV_USE_CUDA
   return Response::SUCCESS;
}

void ComponentBuffer::setBufferSize() {
   PVLayerLoc const *loc     = getLayerLoc();
   int nx                    = loc->nx + getExtendedFlag() * (loc->halo.lt + loc->halo.rt);
   int ny                    = loc->ny + getExtendedFlag() * (loc->halo.dn + loc->halo.up);
   int nf                    = loc->nf;
   int nb                    = loc->nbatch;
   mBufferSize               = nx * ny * nf;
   mBufferSizeAcrossBatch    = mBufferSize * nb;
   mBufferSizeAcrossChannels = mBufferSizeAcrossBatch * mNumChannels;
}

void ComponentBuffer::setReadOnlyPointer() {
   mBufferData.resize(mBufferSizeAcrossChannels);

   mReadOnlyPointer = mBufferData.data();
}

void ComponentBuffer::setReadWritePointer() {
   mReadWritePointer = mBufferData.empty() ? nullptr : mBufferData.data();
}

#ifdef PV_USE_CUDA
void ComponentBuffer::setCudaBuffer() {
   if (mUsingGPUFlag) {
      FatalIf(
            mCudaDevice == nullptr,
            "%s did not receive a SetCudaDevice message.\n",
            getDescription_c());
      std::size_t sizeInBytes = sizeof(float) * (std::size_t)getBufferSizeAcrossChannels();
      mCudaBuffer             = mCudaDevice->createBuffer(sizeInBytes, &getDescription());
   }
}
#endif // PV_USE_CUDA

Response::Status
ComponentBuffer::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mCheckpointFlag and !mBufferLabel.empty()) {
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
            mBufferLabel.c_str());
   }
   return Response::SUCCESS;
}

Response::Status ComponentBuffer::readStateFromCheckpoint(Checkpointer *checkpointer) {
   Response::Status status = BaseObject::readStateFromCheckpoint(checkpointer);
   if (!Response::completed(status)) {
      return status;
   }
   if (mCheckpointFlag and !mBufferLabel.empty()) {
      checkpointer->readNamedCheckpointEntry(std::string(name), mBufferLabel, false);
   }
   return Response::SUCCESS;
}

void ComponentBuffer::checkDimensionsEqual(
      ComponentBuffer const *buffer1,
      ComponentBuffer const *buffer2) {
   checkBatchWidthEqual(buffer1, buffer2);

   PVLayerLoc const *loc1 = buffer1->getLayerLoc();
   PVLayerLoc const *loc2 = buffer2->getLayerLoc();
   bool dimsEqual         = true;
   dimsEqual              = dimsEqual and (loc1->nx == loc2->nx);
   dimsEqual              = dimsEqual and (loc1->ny == loc2->ny);
   dimsEqual              = dimsEqual and (loc1->nf == loc2->nf);

   FatalIf(
         !dimsEqual,
         "%s and %s do not have equal dimensions "
         "(%d-by-%d-by-%d) versus (%d-by-%d-by-%d)\n",
         buffer1->getDescription_c(),
         buffer2->getDescription_c(),
         loc1->nx,
         loc1->ny,
         loc1->nf,
         loc2->nx,
         loc2->ny,
         loc2->nf);
}

void ComponentBuffer::checkDimensionsXYEqual(
      ComponentBuffer const *buffer1,
      ComponentBuffer const *buffer2) {
   checkBatchWidthEqual(buffer1, buffer2);

   PVLayerLoc const *loc1 = buffer1->getLayerLoc();
   PVLayerLoc const *loc2 = buffer2->getLayerLoc();

   FatalIf(
         (loc1->nx != loc2->nx) or (loc1->ny != loc2->ny),
         "%s and %s do not have equal xy dimensions "
         "(%d-by-%d) versus (%d-by-%d)\n",
         buffer1->getDescription_c(),
         buffer2->getDescription_c(),
         loc1->nx,
         loc1->ny,
         loc1->nf,
         loc1->nbatch,
         loc2->nx,
         loc2->ny,
         loc2->nf,
         loc2->nbatch);
}

void ComponentBuffer::checkBatchWidthEqual(
      ComponentBuffer const *buffer1,
      ComponentBuffer const *buffer2) {
   FatalIf(
         buffer1->getLayerLoc()->nbatch != buffer2->getLayerLoc()->nbatch,
         "%s and %s have different batch widths (%d versus %d).\n",
         buffer1->getDescription_c(),
         buffer2->getDescription_c(),
         buffer1->getLayerLoc()->nbatch,
         buffer2->getLayerLoc()->nbatch);
}

void ComponentBuffer::updateBuffer(double simTime, double deltaTime) {
   mTimeLastUpdate = simTime;
#ifdef PV_USE_CUDA
   if (mUsingGPUFlag) {
      updateBufferGPU(simTime, deltaTime);
   }
   else {
      updateBufferCPU(simTime, deltaTime);
   }
#else
   updateBufferCPU(simTime, deltaTime);
#endif // PV_USE_CUDA
}

#ifdef PV_USE_CUDA
void ComponentBuffer::useCuda() {
   FatalIf(
         getDataStructuresAllocatedFlag(),
         "%s cannot set the UsingGPU flag after data allocation.\n",
         getDescription_c());
   mUsingGPUFlag = true;
}

void ComponentBuffer::copyFromCuda() {
   FatalIf(!mCudaBuffer, "%s did not allocate its CudaBuffer.\n", getDescription_c());
   mCudaBuffer->copyFromDevice(mBufferData.data());
}

void ComponentBuffer::copyToCuda() {
   FatalIf(!mCudaBuffer, "%s did not allocate its CudaBuffer.\n", getDescription_c());
   mCudaBuffer->copyToDevice(mBufferData.data());
}

Response::Status ComponentBuffer::copyInitialStateToGPU() {
   copyToCuda();
   return Response::SUCCESS;
}

void ComponentBuffer::updateBufferGPU(double simTime, double deltaTime) {
   Fatal() << "updateGpu for " << getDescription() << " is not implemented\n";
}
#endif // PV_USE_CUDA

} // namespace PV
