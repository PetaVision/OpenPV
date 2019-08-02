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
   auto *objectTable = message->mObjectTable;

   auto *inputLayerNameParam = objectTable->findObject<InputLayerNameParam>(getName());
   FatalIf(
         inputLayerNameParam == nullptr,
         "%s does not have an InputLayerNameParam.\n",
         getDescription_c());
   if (!inputLayerNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   char const *inputLayerName = inputLayerNameParam->getLinkedObjectName();
   mInputController           = objectTable->findObject<InputLayerUpdateController>(inputLayerName);
   FatalIf(
         mInputController == nullptr,
         "%s inputLayerName \"%s\" does not have an InputController.\n",
         getDescription_c(),
         inputLayerName);
   return Response::SUCCESS;
}

bool FilenameParsingLayerUpdateController::needUpdate(double simTime, double deltaTime) const {
   return mInputController->needUpdate(simTime, deltaTime);
}

} // namespace PV
