#include "CheckpointEntryParamsFileWriter.hpp"
#include "checkpointing/CheckpointingMessages.hpp"

namespace PV {

void CheckpointEntryParamsFileWriter::write(
      std::shared_ptr<FileManager const> fileManager, double simTime, bool verifyWritesFlag) const {
   auto message = std::make_shared<WriteParamsFileMessage>(
        fileManager, getName(), WriteParamsFileMessage::WRITE);
   mParamsFileWriter->respond(message);
}

void CheckpointEntryParamsFileWriter::remove(
      std::shared_ptr<FileManager const> fileManager) const {
   auto message = std::make_shared<WriteParamsFileMessage>(
        fileManager, getName(), WriteParamsFileMessage::DELETE);
   mParamsFileWriter->respond(message);
}


} // end namespace PV
