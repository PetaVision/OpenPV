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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

InputRegionActivityBuffer::~InputRegionActivityBuffer() {}

void InputRegionActivityBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ActivityBuffer::initialize(params, defaults, comm);
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
      std::string const &originalLayerName = originalLayerNameParam->getLinkedObjectName();

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
            originalLayerName.c_str());
   }

   if (!mOriginalInput->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   checkDimensionsEqual(mOriginalInput, this);

   mOriginalInput->makeInputRegionsPointer(this);
   return Response::SUCCESS;
}

} // namespace PV
