/*
 * CheckpointerDataInterface.cpp
 *
 *  Created on Feb 22, 2018
 *      Author: Pete Schultz
 */

#include "CheckpointerDataInterface.hpp"

namespace PV {

Response::Status CheckpointerDataInterface::respond(std::shared_ptr<BaseMessage const> message) {
   auto status = Response::NO_ACTION;
   if (message == nullptr) {
      return status;
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<RegisterDataMessage<Checkpointer> const>(message)) {
      return respondRegisterData(castMessage);
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<ReadStateFromCheckpointMessage<Checkpointer> const>(
                     message)) {
      return respondReadStateFromCheckpoint(castMessage);
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<ProcessCheckpointReadMessage const>(message)) {
      return respondProcessCheckpointRead(castMessage);
   }
   else if (
         auto castMessage =
               std::dynamic_pointer_cast<PrepareCheckpointWriteMessage const>(message)) {
      return respondPrepareCheckpointWrite(castMessage);
   }
   else {
      return status;
   }
}

Response::Status CheckpointerDataInterface::respondRegisterData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = registerData(message->mDataRegistry);
   if (!Response::completed(status)) {
      Fatal() << getDescription() << ": registerData failed.\n";
   }
   return status;
}

Response::Status CheckpointerDataInterface::respondReadStateFromCheckpoint(
      std::shared_ptr<ReadStateFromCheckpointMessage<Checkpointer> const> message) {
   return readStateFromCheckpoint(message->mDataRegistry);
}

Response::Status CheckpointerDataInterface::respondProcessCheckpointRead(
      std::shared_ptr<ProcessCheckpointReadMessage const> message) {
   return processCheckpointRead();
}

Response::Status CheckpointerDataInterface::respondPrepareCheckpointWrite(
      std::shared_ptr<PrepareCheckpointWriteMessage const> message) {
   return prepareCheckpointWrite();
}

Response::Status CheckpointerDataInterface::registerData(Checkpointer *checkpointer) {
   if (mMPIBlock) {
      return Response::NO_ACTION;
   }
   else {
      mMPIBlock = checkpointer->getMPIBlock();
      checkpointer->addObserver(this);
      return Response::SUCCESS;
   }
}

} // namespace PV
