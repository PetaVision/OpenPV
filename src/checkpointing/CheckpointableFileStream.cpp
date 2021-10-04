#include "CheckpointableFileStream.hpp"

namespace PV {

CheckpointableFileStream::CheckpointableFileStream(
      string const &path,
      bool newFile,
      Checkpointer *checkpointer,
      string const &objName,
      bool verifyWrites) {
   initialize(path, newFile, checkpointer, objName, verifyWrites);
}

CheckpointableFileStream::CheckpointableFileStream(
      string const &path,
      bool newFile,
      Checkpointer *checkpointer,
      string const &objName) {
   initialize(path, newFile, checkpointer, objName, checkpointer->doesVerifyWrites());
}

void CheckpointableFileStream::initialize(
      string const &path,
      bool newFile,
      Checkpointer *checkpointer,
      string const &objName,
      bool verifyWrites) {
   FatalIf(
         checkpointer->getMPIBlock()->getRank() != 0,
         "CheckpointableFileStream (path \"%s\") called by non-root process.\n",
         path.c_str());
   string fullPath = checkpointer->makeOutputPathFilename(path);

   bool createFile = newFile;
   if (!newFile) {
      // Test if file exists. If not, issue a warning and set createFile to true.
      struct stat statbuf;
      if (PV_stat(fullPath.c_str(), &statbuf) != 0) {
         if (errno == ENOENT) {
            WarnLog().printf(
                  "%s: file \"%s\" does not exist.  Creating new file.\n",
                  getDescription_c(),
                  fullPath.c_str());
            createFile = true;
         }
         else {
            Fatal().printf(
                  "%s: error checking whether file \"%s\" exists: %s \n",
                  getDescription_c(),
                  fullPath.c_str(),
                  strerror(errno));
         }
      }
   }
   if (createFile) {
      char fullPathCopy[fullPath.size() + 1];
      std::memcpy(fullPathCopy, fullPath.c_str(), fullPath.size());
      fullPathCopy[fullPath.size()] = '\0';
      char *dirName                 = dirname(fullPathCopy);
      ensureDirExists(checkpointer->getMPIBlock(), dirName);
      FileStream fileStream(fullPath.c_str(), std::ios_base::out, verifyWrites);
   }

   mObjName = objName;
   setDescription(std::string("CheckpointableFileStream \"") + objName + "\"");
   FileStream::initialize(fullPath.c_str(), std::ios_base::in | std::ios_base::out, verifyWrites);
   CheckpointerDataInterface::initialize();
   updateFilePos();
}

Response::Status CheckpointableFileStream::processCheckpointRead() {
   syncFilePos();
   return Response::SUCCESS;
}

Response::Status CheckpointableFileStream::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = CheckpointerDataInterface::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   checkpointer->registerCheckpointData<long>(
         mObjName, string("FileStreamRead"), &mFileReadPos, (std::size_t)1, false, false);
   checkpointer->registerCheckpointData<long>(
         mObjName, string("FileStreamWrite"), &mFileWritePos, (std::size_t)1, false, false);
   return Response::SUCCESS;
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

void CheckpointableFileStream::setOutPos(long pos, std::ios_base::seekdir seekAnchor) {
   if (seekAnchor != std::ios_base::beg) {
      syncFilePos();
   }
   FileStream::setOutPos(pos, seekAnchor);
}

void CheckpointableFileStream::setOutPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      syncFilePos();
   }
   FileStream::setOutPos(pos, fromBeginning);
   updateFilePos();
}

void CheckpointableFileStream::setInPos(long pos, std::ios_base::seekdir seekAnchor) {
   if (seekAnchor != std::ios_base::beg) {
      syncFilePos();
   }
   FileStream::setInPos(pos, seekAnchor);
}

void CheckpointableFileStream::setInPos(long pos, bool fromBeginning) {
   if (!fromBeginning) {
      syncFilePos();
   }
   FileStream::setInPos(pos, fromBeginning);
   updateFilePos();
}
} // end namespace PV
