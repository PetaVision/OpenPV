/*
 * FilenameParsingLayerUpdateController.cpp
 *
 *  Created on: Nov 20, 2018
 *      Author: peteschultz
 */

#include "FilenameParsingLayerUpdateController.hpp"
#include "components/InputLayerNameParam.hpp"

namespace PV {

FilenameParsingLayerUpdateController::FilenameParsingLayerUpdateController(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

FilenameParsingLayerUpdateController::FilenameParsingLayerUpdateController() {}

void FilenameParsingLayerUpdateController::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   LayerUpdateController::initialize(name, params, comm);
}

FilenameParsingLayerUpdateController::~FilenameParsingLayerUpdateController() {}

void FilenameParsingLayerUpdateController::setObjectType() {
   mObjectType = "FilenameParsingLayerUpdateController";
}

Response::Status FilenameParsingLayerUpdateController::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = LayerUpdateController::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *inputLayerNameParam = message->mHierarchy->lookupByType<InputLayerNameParam>();
   pvAssert(inputLayerNameParam);
   if (!inputLayerNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   ComponentBasedObject *inputObject = nullptr;
   try {
      inputObject = inputLayerNameParam->findLinkedObject(message->mHierarchy);
   } catch (std::invalid_argument &e) {
      Fatal().printf("%s: %s\n", getDescription_c(), e.what());
   }
   FatalIf(
         inputObject == nullptr,
         "%s inputLayerName \"%s\" is not a layer in the column.\n",
         getDescription_c(),
         inputLayerNameParam->getLinkedObjectName());
   mInputController = inputObject->getComponentByType<InputLayerUpdateController>();
   FatalIf(
         mInputController == nullptr,
         "%s inputLayerName \"%s\" does not have an InputController.\n",
         getDescription_c(),
         inputLayerNameParam->getLinkedObjectName());
   return Response::SUCCESS;
}

bool FilenameParsingLayerUpdateController::needUpdate(double simTime, double deltaTime) const {
   return mInputController->needUpdate(simTime, deltaTime);
}

} // namespace PV
