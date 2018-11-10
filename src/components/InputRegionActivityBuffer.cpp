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
      Communicator *comm) {
   initialize(name, params, comm);
}

InputRegionActivityBuffer::~InputRegionActivityBuffer() {}

void InputRegionActivityBuffer::initialize(char const *name, PVParams *params, Communicator *comm) {
   ActivityBuffer::initialize(name, params, comm);
}

void InputRegionActivityBuffer::setObjectType() { mObjectType = "InputRegionActivityBuffer"; }

Response::Status InputRegionActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *hierarchy   = message->mHierarchy;
   int maxIterations = 1; // Limits the depth of the recursion when searching for dependencies.
   auto *originalLayerNameParam =
         hierarchy->lookupByTypeRecursive<OriginalLayerNameParam>(maxIterations);
   FatalIf(
         originalLayerNameParam == nullptr,
         "%s could not find an OriginalLayerName component.\n",
         getDescription_c());
   if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   if (mOriginalInput == nullptr) {
      // Retrieve the original ImageLayer (we don't need to cast it as ImageLayer).
      ComponentBasedObject *originalObject = nullptr;
      try {
         originalObject = originalLayerNameParam->findLinkedObject(hierarchy);
      } catch (std::invalid_argument &e) {
         Fatal().printf("%s: %s\n", getDescription_c(), e.what());
      }
      pvAssert(originalObject);

      // Synchronize margins between original layer and this layer.
      auto *thisGeometry = hierarchy->lookupByTypeRecursive<LayerGeometry>(maxIterations);
      auto *origGeometry = originalObject->getComponentByType<LayerGeometry>();
      LayerGeometry::synchronizeMarginWidths(thisGeometry, origGeometry);

      // Retrieve the original layer's ActivityComponent
      auto *originalActivityComponent = originalObject->getComponentByType<ActivityComponent>();
      FatalIf(
            originalActivityComponent == nullptr,
            "%s could not find an ActivityComponent within %s.\n",
            getDescription_c(),
            originalObject->getName());

      // Retrieve original layer's InputActivityBuffer
      mOriginalInput = originalActivityComponent->getComponentByType<InputActivityBuffer>();
      FatalIf(
            mOriginalInput == nullptr,
            "%s could not find an InputActivityBuffer within %s.\n",
            getDescription_c(),
            originalObject->getName());
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
