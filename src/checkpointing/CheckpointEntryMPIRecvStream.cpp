#include "CheckpointEntryMPIRecvStream.hpp"
#include <sys/stat.h>

namespace PV {

CheckpointEntryMPIRecvStream::CheckpointEntryMPIRecvStream(
      std::string const &name,
      MPIBlock const *mpiBlock,
      MPIRecvStream &mpiRecvStream)
      : CheckpointEntry(name, mpiBlock), mMPIRecvStream(&mpiRecvStream) {}

CheckpointEntryMPIRecvStream::CheckpointEntryMPIRecvStream(
      std::string const &objName,
      std::string const &dataName,
      MPIBlock const *mpiBlock,
      MPIRecvStream &mpiRecvStream)
      : CheckpointEntry(objName, dataName, mpiBlock), mMPIRecvStream(&mpiRecvStream) {}

void CheckpointEntryMPIRecvStream::write(
      std::string const &checkpointDirectory, double simTime, bool verifyWritesFlag) const {
   if (getMPIBlock()->getRank() == 0) {
      long position;
      std::string basePath = generatePath(checkpointDirectory, "");

      position = mMPIRecvStream->getOutPos();
      std::string writeposPath = basePath + "_FileStreamWrite.bin";
      FileStream writePosFileStream(writeposPath.c_str(), std::ios_base::out, verifyWritesFlag);
      writePosFileStream.write(&position, static_cast<long>(sizeof(position)));

      std::string writeposPathTxt = basePath + "_FileStreamWrite.txt";
      FileStream writePosFileStreamTxt(writeposPathTxt.c_str(), std::ios_base::out, verifyWritesFlag);
      writePosFileStreamTxt << position << "\n";

      position = mMPIRecvStream->getInPos();
      std::string readposPath = basePath + "_FileStreamRead.bin";
      FileStream readPosFileStream(readposPath.c_str(), std::ios_base::out, verifyWritesFlag);
      readPosFileStream.write(&position, static_cast<long>(sizeof(position)));

      std::string readposPathTxt = basePath + "_FileStreamRead.txt";
      FileStream readPosFileStreamTxt(readposPathTxt.c_str(), std::ios_base::out, verifyWritesFlag);
      readPosFileStreamTxt << position << "\n";
   }
}

void CheckpointEntryMPIRecvStream::read(
      std::string const &checkpointDirectory, double *simTimePtr) const {
   if (getMPIBlock()->getRank() == 0) {
      long position;
      std::string basePath = generatePath(checkpointDirectory, "");

      std::string writeposPath = basePath + "_FileStreamWrite.bin";
      FileStream writePosFileStream(writeposPath.c_str(), std::ios_base::in, false);
      writePosFileStream.read(&position, static_cast<long>(sizeof(position)));
      mMPIRecvStream->setOutPos(position);

      std::string readposPath = basePath + "_FileStreamRead.bin";
      FileStream readPosFileStream(readposPath.c_str(), std::ios_base::in, false);
      readPosFileStream.read(&position, static_cast<long>(sizeof(position)));
      mMPIRecvStream->setInPos(position);
   }
}

void CheckpointEntryMPIRecvStream::remove(std::string const &checkpointDirectory) const {
   if (getMPIBlock()->getRank() == 0) {
      std::string basePath = generatePath(checkpointDirectory, "");
      std::string removePath = basePath + "_FileStreamWrite.bin";
      unlinkFile(basePath + "_FileStreamWrite.bin");
      unlinkFile(basePath + "_FileStreamWrite.txt");
      unlinkFile(basePath + "_FileStreamRead.bin");
      unlinkFile(basePath + "_FileStreamRead.txt");
   }
}

void CheckpointEntryMPIRecvStream::unlinkFile(std::string const &path) const {
   if (getMPIBlock()->getRank() == 0) {
      struct stat pathStat;
      int statstatus = stat(path.c_str(), &pathStat);
      if (statstatus == 0) {
         int unlinkstatus = unlink(path.c_str());
         FatalIf(
               unlinkstatus != 0, "Failure deleting \"%s\": %s\n", path.c_str(), strerror(errno));
      }
   }
}

} // namespace PV
