/*
 * InputRegionActivityBuffer.cpp
 *
 *  Created on: Aug 30, 2017
 *      Author: pschultz
 */

#include "InputRegionActivityBuffer.hpp"
#include "components/ActivityComponent.hpp"
#include "components/OriginalLayerNameParam.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

InputRegionActivityBuffer::InputRegionActivityBuffer(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

InputRegionActivityBuffer::~InputRegionActivityBuffer() {}

void InputRegionActivityBuffer::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ActivityBuffer::initialize(name, params, comm);
   mCheckpointFlag = false; // Turns off checkpointing
}

void InputRegionActivityBuffer::setObjectType() { mObjectType = "InputRegionActivityBuffer"; }

Response::Status InputRegionActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *objectTable            = message->mObjectTable;
   auto *originalLayerNameParam = objectTable->findObject<OriginalLayerNameParam>(getName());
   FatalIf(
         originalLayerNameParam == nullptr,
         "%s could not find an OriginalLayerName component.\n",
         getDescription_c());
   if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   if (mOriginalInput == nullptr) {
      char const *originalLayerName = originalLayerNameParam->getLinkedObjectName();

      // Synchronize margins between original layer and this layer.
      auto *thisGeometry = objectTable->findObject<LayerGeometry>(getName());
      auto *origGeometry = objectTable->findObject<LayerGeometry>(originalLayerName);
      LayerGeometry::synchronizeMarginWidths(thisGeometry, origGeometry);

      // Retrieve the original layer's activity component
      mOriginalInput = objectTable->findObject<InputActivityBuffer>(originalLayerName);
      FatalIf(
            mOriginalInput == nullptr,
            "%s could not find an InputActivityBuffer within %s.\n",
            getDescription_c(),
            originalLayerName);
   }

   if (!mOriginalInput->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   checkDimensionsEqual(mOriginalInput, this);

   mOriginalInput->makeInputRegionsPointer();
   return Response::SUCCESS;
}

Response::Status InputRegionActivityBuffer::allocateDataStructures() {
   // original layer's InputActivityComponent must allocate first, since setReadOnlyPointer()
   // will copy the original layer's pointer.
   if (!mOriginalInput->getDataStructuresAllocatedFlag()) {
      return Response::POSTPONE;
   }
   return ActivityBuffer::allocateDataStructures();
}

void InputRegionActivityBuffer::setReadOnlyPointer() {
   pvAssert(mOriginalInput);
   mReadOnlyPointer = mOriginalInput->getInputRegionsAllBatchElements();
}

} // namespace PV
