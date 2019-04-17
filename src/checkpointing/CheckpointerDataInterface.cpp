/*
 * CheckpointerDataInterface.cpp
 *
 *  Created on Feb 22, 2018
 *      Author: Pete Schultz
 */

#include "CheckpointerDataInterface.hpp"

namespace PV {

int CheckpointerDataInterface::initialize() { return Observer::initialize(); }

void CheckpointerDataInterface::initMessageActionMap() {
   Observer::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<RegisterDataMessage<Checkpointer> const>(msgptr);
      return respondRegisterData(castMessage);
   };
   mMessageActionMap.emplace("RegisterData", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage =
            std::dynamic_pointer_cast<ReadStateFromCheckpointMessage<Checkpointer> const>(msgptr);
      return respondReadStateFromCheckpoint(castMessage);
   };
   mMessageActionMap.emplace("ReadStateFromCheckpoint", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ProcessCheckpointReadMessage const>(msgptr);
      return respondProcessCheckpointRead(castMessage);
   };
   mMessageActionMap.emplace("ProcessCheckpointRead", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<PrepareCheckpointWriteMessage const>(msgptr);
      return respondPrepareCheckpointWrite(castMessage);
   };
   mMessageActionMap.emplace("PrepareCheckpointWrite", action);
}

Response::Status CheckpointerDataInterface::respondRegisterData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = registerData(message);
   if (!Response::completed(status)) {
      Fatal() << getDescription() << ": registerData failed.\n";
   }
   return status;
}

Response::Status CheckpointerDataInterface::respondReadStateFromCheckpoint(
      std::shared_ptr<ReadStateFromCheckpointMessage<Checkpointer> const> message) {
   if (mInitializeFromCheckpointFlag) {
      return readStateFromCheckpoint(message->mDataRegistry);
   }
   else {
      return Response::NO_ACTION;
   }
}

Response::Status CheckpointerDataInterface::respondProcessCheckpointRead(
      std::shared_ptr<ProcessCheckpointReadMessage const> message) {
   return processCheckpointRead();
}

Response::Status CheckpointerDataInterface::respondPrepareCheckpointWrite(
      std::shared_ptr<PrepareCheckpointWriteMessage const> message) {
   return prepareCheckpointWrite();
}

Response::Status CheckpointerDataInterface::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   if (mMPIBlock) {
      return Response::NO_ACTION;
   }
   else {
      auto *checkpointer = message->mDataRegistry;
      mMPIBlock          = checkpointer->getMPIBlock();
      checkpointer->addObserver(this->getDescription(), this);
      return Response::SUCCESS;
   }
}

} // namespace PV
