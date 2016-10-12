#include "CheckpointableFileStream.hpp"

namespace PV {

CheckpointableFileStream::CheckpointableFileStream(
      char const *path,
      std::ios_base::openmode mode,
      string objName,
      bool verifyWrites) {
   mObjName = objName;
   setOutStream(mFStream);
   openFile(path, mode, verifyWrites);
   updateFilePos();
}

int CheckpointableFileStream::registerData(Secretary *secretary, const string objName) {
   secretary->registerCheckpointData<long>(
         mObjName, std::string("FileStreamRead"), &mFileReadPos, (std::size_t)1, false);
   secretary->registerCheckpointData<long>(
         mObjName, std::string("FileStreamWrite"), &mFileWritePos, (std::size_t)1, false);
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
