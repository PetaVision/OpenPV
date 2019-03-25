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
   mBufferLabel = ""; // Turns off checkpointing
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
      auto hierarchy = message->mHierarchy;
      // Get component holding the original layer name
      int maxIterations = 1; // Limits depth of recursive lookup.
      auto *originalLayerNameParam =
            hierarchy->lookupByTypeRecursive<OriginalLayerNameParam>(maxIterations);
      if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
         return Response::POSTPONE;
      }

      // Retrieve the original HyPerLayer (we don't need to cast it as HyPerLayer).
      ComponentBasedObject *originalObject = nullptr;
      try {
         originalObject = originalLayerNameParam->findLinkedObject(hierarchy);
      } catch (std::invalid_argument &e) {
         Fatal().printf("%s: %s\n", getDescription_c(), e.what());
      }
      pvAssert(originalObject);

      // Retrieve the original layer's ActivityComponent (we don't need to cast it as such)
      auto *activityComponent = originalObject->getComponentByType<ComponentBasedObject>();
      FatalIf(
            activityComponent == nullptr,
            "%s could not find an ActivityComponent within %s.\n",
            getDescription_c(),
            originalObject->getName());

      // Retrieve original layer's InternalStateBuffer
      mOriginalBuffer = activityComponent->getComponentByType<InternalStateBuffer>();
      FatalIf(
            mOriginalBuffer == nullptr,
            "%s could not find an InternalStateBuffer within %s.\n",
            getDescription_c(),
            originalObject->getName());
   }
   if (!mOriginalBuffer->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   checkDimensionsEqual(mOriginalBuffer, this);

   return Response::SUCCESS;
}

void CloneInternalStateBuffer::setReadOnlyPointer() {
   pvAssert(mBufferData.empty()); // nothing else should have allocated this
   pvAssert(mOriginalBuffer); // set in communicateInitInfo
   mReadOnlyPointer = mOriginalBuffer->getBufferData();
}

} // namespace PV
