/*
 * InternalStateBuffer.cpp
 *
 *  Created on: Sep 6, 2018
 *      Author: Pete Schultz
 */

#include "InternalStateBuffer.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

InternalStateBuffer::InternalStateBuffer(char const *name, HyPerCol *hc) { initialize(name, hc); }

InternalStateBuffer::~InternalStateBuffer() { free(mInitVTypeString); }

int InternalStateBuffer::initialize(char const *name, HyPerCol *hc) {
   int status    = BufferComponent::initialize(name, hc);
   mExtendedFlag = false;
   mBufferLabel  = "V";
   return status;
}

void InternalStateBuffer::setObjectType() { mObjectType = "InternalStateBuffer"; }

int InternalStateBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_InitVType(ioFlag);
   return PV_SUCCESS;
}

void InternalStateBuffer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamString(
         ioFlag,
         name,
         "InitVType",
         &mInitVTypeString,
         BaseInitV::mDefaultInitV.data(),
         true /*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      BaseObject *object = Factory::instance()->createByKeyword(mInitVTypeString, name, parent);
      mInitVObject       = dynamic_cast<BaseInitV *>(object);
      if (mInitVObject == nullptr) {
         ErrorLog().printf("%s: unable to create InitV object\n", getDescription_c());
         abort();
      }
   }
   if (mInitVObject != nullptr) {
      mInitVObject->ioParamsFillGroup(ioFlag);
   }
}

Response::Status InternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   Response::Status status = BufferComponent::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mInputBuffer = message->mHierarchy->lookupByType<LayerInputBuffer>();
   FatalIf(
         mInputBuffer == nullptr,
         "%s could not find a LayerInputBuffer component.\n",
         getDescription_c());
   checkDimensions(mInputBuffer->getLayerLoc(), getLayerLoc());
   return Response::SUCCESS;
}

void InternalStateBuffer::checkDimensions(PVLayerLoc const *inLoc, PVLayerLoc const *outLoc) const {
   checkDimension(inLoc->nx, outLoc->nx, "nx");
   checkDimension(inLoc->ny, outLoc->ny, "ny");
   checkDimension(inLoc->nf, outLoc->nf, "nf");
   checkDimension(inLoc->nbatch, outLoc->nbatch, "nbatch");
}

void InternalStateBuffer::checkDimension(int gSynSize, int internalStateSize, char const *fieldname)
      const {
   FatalIf(
         gSynSize != internalStateSize,
         "%s and %s do not have the same %s (%d versus %d)\n",
         mInputBuffer->getDescription(),
         getDescription(),
         fieldname,
         gSynSize,
         internalStateSize);
}

Response::Status InternalStateBuffer::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BufferComponent::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mInitVObject != nullptr) {
      mInitVObject->respond(message);
   }
   return Response::SUCCESS;
}

Response::Status
InternalStateBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   if (mInitVObject != nullptr) {
      mInitVObject->calcV(mBufferData.data(), getLayerLoc());
   }
   return Response::SUCCESS;
}

void InternalStateBuffer::updateBuffer(double simTime, double deltaTime) {
   float const *gSynHead = mInputBuffer->getBufferData();
   float *V              = mBufferData.data();

   int numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   if (mInputBuffer->getNumChannels() == 1) {
      float const *gSynExc = mInputBuffer->getChannelData(CHANNEL_EXC);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int k = 0; k < numNeuronsAcrossBatch; k++) {
         V[k] = gSynExc[k];
      }
   }
   else {
      float const *gSynExc = mInputBuffer->getChannelData(CHANNEL_EXC);
      float const *gSynInh = mInputBuffer->getChannelData(CHANNEL_INH);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int k = 0; k < numNeuronsAcrossBatch; k++) {
         V[k] = gSynExc[k] - gSynInh[k];
      }
   }
}

} // namespace PV
