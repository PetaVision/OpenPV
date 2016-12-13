#include "CheckpointableFileStream.hpp"

namespace PV {

CheckpointableFileStream::CheckpointableFileStream(
      char const *path,
      std::ios_base::openmode mode,
      string objName,
      bool verifyWrites) {
   mObjName = objName;
   setDescription();
   setOutStream(mFStream);
   openFile(path, mode, verifyWrites);
   updateFilePos();
}

void CheckpointableFileStream::setDescription() {
   description = "CheckpointableFileStream \"";
   description.append(mObjName).append("\"");
}

int CheckpointableFileStream::respond(std::shared_ptr<BaseMessage const> message) {
   if (message == nullptr) {
      return PV_SUCCESS;
   }
   else if (
         ProcessCheckpointReadMessage const *castMessage =
               dynamic_cast<ProcessCheckpointReadMessage const *>(message.get())) {
      return respondProcessCheckpointRead(castMessage);
   }
}

int CheckpointableFileStream::respondProcessCheckpointRead(
      ProcessCheckpointReadMessage const *message) {
   syncFilePos();
   return PV_SUCCESS;
}

int CheckpointableFileStream::registerData(Checkpointer *checkpointer, const string objName) {
   checkpointer->registerCheckpointData<long>(
         mObjName, std::string("FileStreamRead"), &mFileReadPos, (std::size_t)1, false);
   checkpointer->registerCheckpointData<long>(
         mObjName, std::string("FileStreamWrite"), &mFileWritePos, (std::size_t)1, false);
   checkpointer->addObserver(this, BaseMessage{});
   return PV_SUCCESS;
}

// When restoring from checkpoint, the variables mFileReadPos and mFileWritePos
// will be modified. This method checks to see if this has happened, and seeks
// to the correct location in the files.
void CheckpointableFileStream::syncFilePos() {
   if (mFileReadPos != getInPos()) {
      setInPos(mFileReadPos, true);
   }
   if (mFileWritePos != getOutPos()) {
      setOutPos(mFileWritePos, true);
   }
}

void CheckpointableFileStream::updateFilePos() {
   mFileReadPos  = getInPos();
   mFileWritePos = getOutPos();
}

void CheckpointableFileStream::write(void const *data, long length) {
   syncFilePos();
   FileStream::write(data, length);
   updateFilePos();
}

void CheckpointableFileStream::read(void *data, long length) {
   syncFilePos();
   FileStream::read(data, length);
   updateFilePos();
}

void CheckpointableFileStream::setOutPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      syncFilePos();
   }
   FileStream::setOutPos(pos, fromBeginning);
   updateFilePos();
}

void CheckpointableFileStream::setInPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      syncFilePos();
   }
   FileStream::setInPos(pos, fromBeginning);
   updateFilePos();
}
}
