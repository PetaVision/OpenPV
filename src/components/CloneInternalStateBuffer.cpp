/*
 * CloneInternalStateBuffer.cpp
 *
 *  Created on: Aug 15, 2013
 *      Author: pschultz
 */

#include "CloneInternalStateBuffer.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

CloneInternalStateBuffer::CloneInternalStateBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

CloneInternalStateBuffer::~CloneInternalStateBuffer() {}

void CloneInternalStateBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   InternalStateBuffer::initialize(name, params, comm);
   mCheckpointFlag = false; // Turns off checkpointing
}

void CloneInternalStateBuffer::setObjectType() { mObjectType = "CloneInternalStateBuffer"; }

void CloneInternalStateBuffer::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "InitVType");
   }
}

Response::Status CloneInternalStateBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = InternalStateBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   if (mOriginalBuffer == nullptr) {
      auto *objectTable            = message->mObjectTable;
      auto *originalLayerNameParam = objectTable->findObject<OriginalLayerNameParam>(getName());
      if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
         return Response::POSTPONE;
      }
      FatalIf(
            originalLayerNameParam == nullptr,
            "%s could not find an OriginalLayerNameParam.\n",
            getDescription_c());

      // Retrieve original layer's InternalStateBuffer
      char const *originalLayerName = originalLayerNameParam->getLinkedObjectName();
      mOriginalBuffer = objectTable->findObject<InternalStateBuffer>(originalLayerName);
      FatalIf(
            mOriginalBuffer == nullptr,
            "%s could not find an InternalStateBuffer within %s.\n",
            getDescription_c(),
            originalLayerName);
   }
   if (!mOriginalBuffer->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   checkDimensionsEqual(mOriginalBuffer, this);

   return Response::SUCCESS;
}

Response::Status CloneInternalStateBuffer::allocateDataStructures() {
   if (!mOriginalBuffer->getDataStructuresAllocatedFlag()) {
      return Response::POSTPONE;
   }
   else {
      return InternalStateBuffer::allocateDataStructures();
   }
}

void CloneInternalStateBuffer::setReadOnlyPointer() {
   pvAssert(mBufferData.empty()); // nothing else should have allocated this
   pvAssert(mOriginalBuffer); // set in communicateInitInfo
   mReadOnlyPointer = mOriginalBuffer->getBufferData();
}

} // namespace PV
